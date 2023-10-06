import cuda_kernels
import math
from numba import cuda
import numpy as np
import time
from utility_functions import blockPrint, enablePrint, print_progress


def AveragesOnShellsUsingLogicBCuda(inc, retofRR, retofRI, n1ofR, n2ofR, kXofR, kYofR, kZofR,
                                    NumAtEachR, Thresh, RMax):
    print('This loop will go to ' + str(RMax) + '\n')
    NumAtEachRMax = NumAtEachR[-1]
    retofROutR = np.zeros((inc + 1, NumAtEachRMax))
    retofROutI = np.zeros((inc + 1, NumAtEachRMax))
    n1ofROut = np.zeros((inc + 1, NumAtEachRMax))
    n2ofROut = np.zeros((inc + 1, NumAtEachRMax))
    NumAtROut = np.zeros((inc + 1, NumAtEachRMax))

    print("shape of retofROutR is ", np.shape(retofROutR))
    NumAtEachRMaxCuda = 15871

    retofROutR[0, 0] = retofRR[0, 0]
    retofROutI[0, 0] = retofRI[0, 0]
    n1ofROut[0, 0] = n1ofR[0, 0]
    n2ofROut[0, 0] = n2ofR[0, 0]

    # Load all data into GPU memory
    # Need to convert data to contiguous arrays
    retofRR_global_mem = cuda.to_device(np.ascontiguousarray(retofRR, dtype=np.float32))
    retofRI_global_mem = cuda.to_device(np.ascontiguousarray(retofRI, dtype=np.float32))
    n1ofR_global_mem = cuda.to_device(np.ascontiguousarray(n1ofR, dtype=np.float32))
    n2ofR_global_mem = cuda.to_device(np.ascontiguousarray(n2ofR, dtype=np.float32))
    enablePrint()
    for r in range(1, RMax + 1):
        if ((r - 1) % 5) == 0:
            print(r)
        NumOnSurf = int(NumAtEachR[r])
        kXNow = kXofR[r][:NumOnSurf]
        kYNow = kYofR[r][:NumOnSurf]
        kZNow = kZofR[r][:NumOnSurf]  # Vectors

        ## Progress bar
        print_progress(r, RMax)

        NumLoops = 1 + int(NumOnSurf * NumOnSurf / NumAtEachRMaxCuda / NumAtEachRMaxCuda)  # kicks in at r=50
        Stride = int(NumOnSurf / NumLoops)

        blockPrint()
        print("NumOnSurf:NumLoops:Stride -  ", NumOnSurf, ":", NumLoops, ":", Stride)
        stream = cuda.stream()
        for jLoop in range(NumLoops):

            Start = jLoop * Stride
            End = Start + Stride
            if jLoop == (NumLoops - 1):
                End = NumOnSurf
            InnerLogicCuda_start = time.time()
            NumAtROutPre_global_mem = AveragesOnShellsInnerLogicKernelCuda(kXNow, kYNow, kZNow,
                                                                           retofRR_global_mem,
                                                                           retofRI_global_mem,
                                                                           n1ofR_global_mem,
                                                                           n2ofR_global_mem,
                                                                           NumOnSurf,
                                                                           Thresh,
                                                                           Start,
                                                                           End,
                                                                           r)
            print("Time to complete InnerLogicKernelCuda is ", time.time() - InnerLogicCuda_start)
            startTime = time.time()
            LogicCCuda_start = startTime

            reduced = AveragesOnShellsInnerLogicCCuda(
                retofRR_global_mem,
                retofRI_global_mem,
                n1ofR_global_mem,
                n2ofR_global_mem,
                NumAtROutPre_global_mem,
                End,
                Start,
                NumOnSurf,
                r)
            stream = cuda.stream()
            cuda.synchronize()
            print("\nTime to complete AveragesOnShellsInnerLogicCCuda is ", time.time() - LogicCCuda_start)

            retofROutR[r][Start:End] = reduced[0]
            retofROutI[r][Start:End] = reduced[1]
            n1ofROut[r][Start:End] = reduced[2]
            n2ofROut[r][Start:End] = reduced[3]

            sum_start = time.time()
            print("Dimensions of NumAtROutPre_global_mem are ", NumAtROutPre_global_mem.shape)
            NumAtROut[r][Start:End] = cuda_kernels.sum_rows(NumAtROutPre_global_mem, Start, End)
            print("Time to compute sum_rows on GPU: ", time.time() - sum_start)

    return retofROutR, retofROutI, n1ofROut, n2ofROut, NumAtROut


# For a given shell, this function returns whether a pair are close or not

def AveragesOnShellsInnerLogicKernelCuda(
        kXNow, kYNow, kZNow,
        retofRR_global_mem,
        retofRI_global_mem,
        n1ofR_global_mem,
        n2ofR_global_mem,
        NumOnSurf,
        Thresh,
        Start,
        End,
        r):
    stream = cuda.stream()
    NumAtROutPre_global_mem = cuda.device_array((NumOnSurf, End - Start))
    Prod11_global_mem = cuda.device_array(NumOnSurf, stream=stream, dtype=np.float32)

    # Set threads per block and blocks per grid
    threadsperblock = (1024, 1, 1)
    blockspergrid_x = int(math.ceil(kXNow[:NumOnSurf].shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, 1, 1)
    start_cuda = time.time()

    # Kernel 1, calculate Prod11
    cuda_kernels.cuda_calcProd11[blockspergrid, threadsperblock, stream](
        kXNow, kYNow, kZNow,
        Prod11_global_mem,
        NumOnSurf,
        r)

    stream.synchronize()
    # Kernel 2, calculate Inner2
    cuda_kernels.cuda_calcInner2[blockspergrid, threadsperblock, stream](
        kXNow, kYNow, kZNow,
        Prod11_global_mem,
        NumAtROutPre_global_mem,
        End,
        Start,
        Thresh,
        NumOnSurf,
        r)
    stream.synchronize()
    end_cuda = time.time()
    print("\nCUDA inner calculations completed in " + str(end_cuda - start_cuda) + ".")
    return NumAtROutPre_global_mem


def AveragesOnShellsInnerLogicCCuda(retNowR_global_mem, retNowI_global_mem, n1ofR_global_mem, n2ofR_global_mem,
                                    NumAtROutPre_global_mem, End, Start, NumOnSurf, r):
    threadsperblock = (1024, 1, 1)
    blockspergrid_x = int(math.ceil(retNowR_global_mem[r][:NumOnSurf].shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, 1, 1)
    # set up stream
    stream = cuda.stream()
    reduced_global_mem = cuda.device_array((4, (End - Start)))
    print("Shape of NumAtROutPre_global_mem is ", NumAtROutPre_global_mem.shape)
    print("Shape of reduced_global_mem is ", reduced_global_mem.shape)
    filter_time = time.time()
    cuda_kernels.filter_and_sum[threadsperblock, blockspergrid](retNowR_global_mem, retNowI_global_mem,
                                                                n1ofR_global_mem, n2ofR_global_mem,
                                                                NumAtROutPre_global_mem, reduced_global_mem, End, Start,
                                                                NumOnSurf, r)
    stream.synchronize()
    print("Time to complete filter_and_sum is ", time.time() - filter_time)
    reduced_start = time.time()
    reduced = reduced_global_mem.copy_to_host()
    print("Time to transfer reduced to host is ", time.time() - reduced_start)

    return reduced


# From Analysis script
def threshold_binarize_array_cuda(dataarray, FSCCutoff, ThresholdForSphericity, highpassfilter):
    # Thresholds
    cutoff_fsc = float(FSCCutoff)
    cutoff_binarize = float(ThresholdForSphericity)
    min_cutoff = min(cutoff_fsc, cutoff_binarize)

    # Coordinates
    center = (dataarray.shape[0] / 2, dataarray.shape[1] / 2, dataarray.shape[2] / 2)

    # Fill up new np array
    boxsize = dataarray.shape[0]
    outarraythresholded = np.zeros((boxsize,) * 3)
    outarraythresholdedbinarized = np.zeros((boxsize,) * 3)

    points_array = cuda_kernels.calcDistance(boxsize, center)

    start_time = time.time()
    sorted_indexes = np.lexsort((points_array[:, 1], points_array[:, 0]))
    points_array = points_array[sorted_indexes]
    print("Sorting time is ", time.time() - start_time)

    memory_inmrc_thresholded = np.copy(dataarray)
    memory_inmrc_thresholdedbinarized = np.copy(dataarray)
    outarraythresholded, outarraythresholdedbinarized = cuda_kernels.calcNeighbors(points_array, center, cutoff_fsc,
                                                                                   cutoff_binarize, highpassfilter,
                                                                                   min_cutoff, dataarray,
                                                                                   memory_inmrc_thresholded,
                                                                                   memory_inmrc_thresholdedbinarized,
                                                                                   outarraythresholded,
                                                                                   outarraythresholdedbinarized)

    return outarraythresholded, outarraythresholdedbinarized
