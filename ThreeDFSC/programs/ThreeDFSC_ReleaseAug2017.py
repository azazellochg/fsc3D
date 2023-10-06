#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Anaconda 3 required
# This program is  ThreeDFSC_ReleaseFeb2017.py
# A conical resolution program written by P. R. Baldwin in November 2016
# Downloaded from https://github.com/nysbc/Anisotropy
# ThreeDFSC_ReleaseJul2017.py
#                                 HalfMap1.mrc    HalfMap2.mrc   OutputStringLabel       A/pixel DeltaTheta                                       
#
# Creates       ResEMOutresultAve+OutputStringLabel.csv (which is the usual FSC)
#                       ResEMOut+OutputStringLabel.hdf which is the 3D FSC file ResEMR (the real part of the cccs) 
#                       Plots+OutputStringLabel.jpg which is the slices along x, y, z 
#
# This requires the existence of numba, but not numbapro
#
# The functions need to be kept separated so that the precompiler can
# notice the @jit decorations and  precompile the code,
#
# Uses mrcfile 1.0.0 by Colin Palmer (https://github.com/ccpem/mrcfile)
# 
# For Phil: Line 547 for jSurf in range(Num2Surf+1): ### Fixed the problem!

from sys import argv
import csv
import time
import os
import numpy as np
from math import *
from numba import *
import matplotlib.pyplot as plt
import mrcfile
import click

from utility_functions import blockPrint, enablePrint
import cuda_functions


# %%       Section -1 Function Definitions

@jit(nopython=True)
def ExtractAxes(f):
    [nx, ny, nz] = f.shape

    nx2 = nx // 2
    ny2 = ny // 2
    nz2 = nz // 2

    xf = np.zeros(nx2 + 1)
    yf = np.zeros(ny2 + 1)
    zf = np.zeros(nz2 + 1)

    for ix in range(nx2):
        xf[ix] = f[ix + nx2, ny2, nz2]

    for iy in range(ny2):
        yf[iy] = f[nx2, iy + ny2, nz2]

    for iz in range(nz2):
        zf[iz] = f[nx2, ny2, iz + nz2]

    return xf, yf, zf


# %%      Section -1 Function Definitions

@jit(nopython=True)
def AddAxes(f, jDir, Val):
    [nx, ny, nz] = f.shape

    fOut = f.copy()
    nx2 = nx // 2
    ny2 = ny // 2
    nz2 = nz // 2

    if jDir == 0:
        fOut[:, ny2 - 1, nz2 - 1] = Val

    if jDir == 1:
        fOut[nx2 - 1, :, nz2 - 1] = Val

    if jDir == 2:
        fOut[nx2 - 1, ny2 - 1, :] = Val

    return fOut


@jit(nopython=True)
def ZeroPad(nx, ny, nz, fT, gT):
    fp = np.zeros((nx + 2, ny, nz))
    gp = np.zeros((nx + 2, ny, nz))

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                fp[ix, iy, iz] = fT[ix, iy, iz]
                gp[ix, iy, iz] = gT[ix, iy, iz]
    return fp, gp


@jit(nopython=True)
def FFTArray2Real(nx, ny, nz, F):
    d1 = np.zeros((nx + 4, ny, nz))  # 36,32,32

    for ix in range(0, nx + 3, 2):
        ixover2 = ix // 2
        for iy in range(ny):
            for iz in range(nz):
                FNow = F[ixover2, iy, iz]
                d1[ix][iy][iz] = FNow.real
                d1[ix + 1][iy][iz] = FNow.imag
    return d1


@jit(nopython=True)
def CreateFTLikeOutputs(inc, nx, ny, nz, ToBeAveraged, nx2, ny2, nz2, dx2, dy2, dz2):
    ret = np.zeros(inc + 1, dtype=np.float64)
    lr = np.zeros(inc + 1, dtype=np.float64)
    for iz in range(nz):
        kz = iz
        if iz > nz2:
            kz = iz - nz  # This is the actual value kz, can be neg
        argz = float(kz * kz) * dz2
        for iy in range(ny):
            ky = iy  # This is the actual value ky, can be neg
            if iy > ny2:
                ky = iy - ny
            argy = argz + float(ky * ky) * dy2
            for ix in range(0, nx, 2):
                if ix == 0 and kz < 0 and ky < 0:
                    continue
                argx = 0.5 * np.sqrt(argy + float(ix * ix) * 0.25 * dx2)
                r = int(round(inc * 2 * argx))
                if r <= inc:
                    # ii = ix + (iy  + iz * ny)* lsd2;
                    ret[r] += ToBeAveraged[ix, iy, iz]
                    lr[r] += 2.0  # Number of Values

    return ret, lr


@jit(nopython=True)
def CreateFSCOutputs(inc, nx, ny, nz, d1, d2, nx2, ny2, nz2, dx2, dy2, dz2):
    ret = np.zeros(inc + 1, dtype=np.float64)
    n1 = np.zeros(inc + 1, dtype=np.float64)
    n2 = np.zeros(inc + 1, dtype=np.float64)
    lr = np.zeros(inc + 1, dtype=np.float64)
    for iz in range(nz):
        kz = iz
        if iz > nz2:
            kz = iz - nz  # This is the actual value kz, can be neg
        argz = float(kz * kz) * dz2
        for iy in range(ny):
            ky = iy  # This is the actual value ky, can be neg
            if iy > ny2:
                ky = iy - ny
            argy = argz + float(ky * ky) * dy2
            for ix in range(0, nx, 2):
                if ix == 0 and kz < 0 and ky < 0:
                    continue
                argx = 0.5 * np.sqrt(argy + float(ix * ix) * 0.25 * dx2)
                r = int(round(inc * 2 * argx))
                if r <= inc:
                    ret[r] += d1[ix, iy, iz] * d2[ix, iy, iz]
                    ret[r] += d1[ix + 1, iy, iz] * d2[ix + 1, iy, iz]
                    n1[r] += d1[ix, iy, iz] * d1[ix, iy, iz]
                    n1[r] += d1[ix + 1, iy, iz] * d1[ix + 1, iy, iz]
                    n2[r] += d2[ix, iy, iz] * d2[ix, iy, iz]
                    n2[r] += d2[ix + 1, iy, iz] * d2[ix + 1, iy, iz]
                    lr[r] += 2.0  # Number of Values

    return ret, n1, n2, lr


@jit(nopython=True)
def createFSCarrays(nx, ny, nz, lsd2, lr, inc, dx2, dy2, dz2, d1, d2, nx2, ny2, nz2):
    """ Find values of product at individual points, organized on shells in FS """
    lrMaxOver2 = int(lr[-1] // 2)

    kXofR = np.zeros((inc + 1, lrMaxOver2), dtype=np.int32)
    kYofR = np.zeros((inc + 1, lrMaxOver2), dtype=np.int32)
    kZofR = np.zeros((inc + 1, lrMaxOver2), dtype=np.int32)
    retofRR = np.zeros((inc + 1, lrMaxOver2), dtype=np.float64)
    retofRI = np.zeros((inc + 1, lrMaxOver2), dtype=np.float64)
    n1ofR = np.zeros((inc + 1, lrMaxOver2), dtype=np.float64)
    n2ofR = np.zeros((inc + 1, lrMaxOver2), dtype=np.float64)

    NumAtEachR = np.zeros(inc + 1, dtype=np.int32)

    rmax = 0
    for iz in range(nz):
        kz = iz
        if iz > nz2:
            kz = iz - nz  # This is the actual value kz, can be neg
        argz = float(kz * kz) * dz2
        for iy in range(ny):
            ky = iy  # This is the actual value ky, can be neg
            if iy > ny2:
                ky = iy - ny
            argy = argz + float(ky * ky) * dy2
            for ix in range(0, nx, 2):
                if ix == 0 and ky < 0:
                    continue
                if ix == 0 and ky == 0 and kz < 0:
                    continue
                kx = float(ix) / 2.0
                argx = 0.5 * np.sqrt(argy + kx * kx * dx2)
                r = int(round(inc * 2 * argx))
                if r <= inc:
                    NumAtEachR[r] += 1
                    LastInd = NumAtEachR[r] - 1
                    kXofR[r, LastInd] = int(round(kx))
                    kYofR[r, LastInd] = ky
                    kZofR[r, LastInd] = kz
                    retrRNow = d1[ix, iy, iz] * d2[ix, iy, iz]
                    retrRNow += d1[ix + 1, iy, iz] * d2[ix + 1, iy, iz]
                    retrINow = d1[ix, iy, iz] * d2[ix + 1, iy, iz]
                    retrINow -= d1[ix + 1, iy, iz] * d2[ix, iy, iz]
                    n1rNow = d1[ix, iy, iz] * d1[ix, iy, iz]
                    n1rNow += d1[ix + 1, iy, iz] * d1[ix + 1, iy, iz]
                    n2rNow = d2[ix, iy, iz] * d2[ix, iy, iz]
                    n2rNow += d2[ix + 1, iy, iz] * d2[ix + 1, iy, iz]
                    retofRR[r, LastInd] = retrRNow
                    retofRI[r, LastInd] = retrINow
                    n1ofR[r, LastInd] = n1rNow
                    n2ofR[r, LastInd] = n2rNow

                    if r > rmax:
                        rmax = r

    return kXofR, kYofR, kZofR, retofRR, retofRI, n1ofR, n2ofR, NumAtEachR


@jit(nopython=True)
def createFTarrays(nx, ny, nz, lsd2, lr, inc, dx2, dy2, dz2, dcH, dFPower, nx2, ny2, nz2):
    """ Find values of product at individual points, organized on shells in FS """
    lrMaxOver2 = int(lr[-1] // 2)

    kXofR = np.zeros((inc + 1, lrMaxOver2), dtype=np.int32)
    kYofR = np.zeros((inc + 1, lrMaxOver2), dtype=np.int32)
    kZofR = np.zeros((inc + 1, lrMaxOver2), dtype=np.int32)
    retcH = np.zeros((inc + 1, lrMaxOver2), dtype=np.float64)
    retFT = np.zeros((inc + 1, lrMaxOver2), dtype=np.float64)
    n12ofR = np.zeros((inc + 1, lrMaxOver2), dtype=np.float64)

    NumAtEachR = np.zeros(inc + 1, dtype=np.int32)
    #
    rmax = 0
    for iz in range(nz):
        kz = iz
        if iz > nz2:
            kz = iz - nz  # This is the actual value kz, can be neg
        argz = float(kz * kz) * dz2
        for iy in range(ny):
            ky = iy  # This is the actual value ky, can be neg
            if iy > ny2:
                ky = iy - ny
            argy = argz + float(ky * ky) * dy2
            for ix in range(0, nx, 2):
                if ix == 0 and ky < 0:
                    continue
                if ix == 0 and ky == 0 and kz < 0:
                    continue
                kx = float(ix) / 2.0
                argx = 0.5 * np.sqrt(argy + kx * kx * dx2)
                r = int(round(inc * 2 * argx))
                if r <= inc:
                    NumAtEachR[r] += 1
                    LastInd = NumAtEachR[r] - 1
                    kXofR[r, LastInd] = int(round(kx))
                    kYofR[r, LastInd] = ky
                    kZofR[r, LastInd] = kz
                    retcHNow = dcH[ix, iy, iz]
                    retFTNow = dFPower[ix, iy, iz]
                    n12rNow = 1
                    retcH[r, LastInd] = retcHNow
                    retFT[r, LastInd] = retFTNow
                    n12ofR[r, LastInd] = n12rNow

                    if r > rmax:
                        rmax = r
    print(rmax)
    return kXofR, kYofR, kZofR, retcH, retFT, n12ofR


@jit(nopython=True)
def AveragesOnShellsInnerLogicKernelnonCuda(kXNow, kYNow, kZNow, NumOnSurf, Thresh, Start, End):
    #       NumAtROutPre has dimensions NumOnSurf by End-Start
    #       Each element NumAtROutPre[m,n]  denotes whether m is close to Start+n

    NumAtROutPre = np.zeros((NumOnSurf, End - Start), dtype=np.int32)

    Thresh2 = Thresh * Thresh
    for jSurf1 in range(NumOnSurf):
        kX1 = kXNow[jSurf1]
        kY1 = kYNow[jSurf1]
        kZ1 = kZNow[jSurf1]  # Single Values
        Prod11 = kX1 * kX1 + kY1 * kY1 + kZ1 * kZ1
        if Prod11 == 0:
            continue
        for jSurf2 in range(End - Start):  # labels kX, etc
            kX2 = kXNow[jSurf2 + Start]
            kY2 = kYNow[jSurf2 + Start]
            kZ2 = kZNow[jSurf2 + Start]  # Single Values
            Prod12 = kX1 * kX2 + kY1 * kY2 + kZ1 * kZ2
            Prod22 = kX2 * kX2 + kY2 * kY2 + kZ2 * kZ2
            if Prod22 == 0:
                continue
            Inner2 = Prod12 * Prod12 / (Prod11 * Prod22)
            if Inner2 > Thresh2:  # Then angle is sufficiently small
                NumAtROutPre[jSurf1, jSurf2] = 1

    return NumAtROutPre


@jit(nopython=True)
def AveragesOnShellsInnerLogicC(retNowR, retNowI, n1Now, n2Now, Start, End, NumAtROutPre):
    NumNow = End - Start
    retofROutRPre = np.zeros(NumNow)
    retofROutIPre = np.zeros(NumNow)
    n1ofROutPre = np.zeros(NumNow)
    n2ofROutPre = np.zeros(NumNow)

    for jSurf1 in range(NumNow):
        MultVec = NumAtROutPre[:, jSurf1]  # NumAtROutPre has shape 15871 by 7936
        GoodInds = np.where(MultVec)[0]  # MultVec has shape 15871

        retofROutRPre[jSurf1] = np.sum(retNowR[GoodInds])
        retofROutIPre[jSurf1] = np.sum(retNowI[GoodInds])
        n1ofROutPre[jSurf1] = np.sum(n1Now[GoodInds])
        n2ofROutPre[jSurf1] = np.sum(n2Now[GoodInds])

    return retofROutRPre, retofROutIPre, n1ofROutPre, n2ofROutPre


def AveragesOnShellsUsingLogicB(inc, retofRR, retofRI, n1ofR, n2ofR, kXofR, kYofR, kZofR, NumAtEachR, Thresh, RMax):
    print('This loop will go to ' + str(RMax) + '\n')
    NumAtEachRMax = NumAtEachR[-1]
    retofROutR = np.zeros((inc + 1, NumAtEachRMax))
    retofROutI = np.zeros((inc + 1, NumAtEachRMax))
    n1ofROut = np.zeros((inc + 1, NumAtEachRMax))
    n2ofROut = np.zeros((inc + 1, NumAtEachRMax))
    NumAtROut = np.zeros((inc + 1, NumAtEachRMax))
    NumAtEachRMaxCuda = 15871

    retofROutR[0, 0] = retofRR[0, 0]
    retofROutI[0, 0] = retofRI[0, 0]
    n1ofROut[0, 0] = n1ofR[0, 0]
    n2ofROut[0, 0] = n2ofR[0, 0]

    enablePrint()
    with click.progressbar(length=(RMax ** 6)) as bar:
        for r in range(1, RMax + 1):
            NumOnSurf = int(NumAtEachR[r])
            kXNow = kXofR[r][:NumOnSurf]
            kYNow = kYofR[r][:NumOnSurf]
            kZNow = kZofR[r][:NumOnSurf]  # Vectors
            retNowR = retofRR[r][:NumOnSurf]
            retNowI = retofRI[r][:NumOnSurf]
            n1Now = n1ofR[r][:NumOnSurf]
            n2Now = n2ofR[r][:NumOnSurf]  # for given

            ## Progress bar
            bar.update((r ** 6) - ((r - 1) ** 6))

            NumLoops = 1 + int(NumOnSurf * NumOnSurf / NumAtEachRMaxCuda / NumAtEachRMaxCuda)  # kicks in at r=50
            Stride = int(NumOnSurf / NumLoops)
            for jLoop in range(NumLoops):

                Start = jLoop * Stride
                End = Start + Stride
                if jLoop == (NumLoops - 1):
                    End = NumOnSurf
                NumAtROutPre = AveragesOnShellsInnerLogicKernelnonCuda(kXNow, kYNow, kZNow, NumOnSurf, Thresh, Start,
                                                                       End)
                retofROutRPre, retofROutIPre, n1ofROutPre, n2ofROutPre = AveragesOnShellsInnerLogicC(retNowR, retNowI,
                                                                                                       n1Now, n2Now,
                                                                                                       Start, End,
                                                                                                       NumAtROutPre)
                retofROutR[r][Start:End] = retofROutRPre
                retofROutI[r][Start:End] = retofROutIPre
                n1ofROut[r][Start:End] = n1ofROutPre
                n2ofROut[r][Start:End] = n2ofROutPre
                qq = np.sum(NumAtROutPre, axis=0)
                NumAtROut[r][Start:End] = qq

    blockPrint()
    return retofROutR, retofROutI, n1ofROut, n2ofROut, NumAtROut


@jit(nopython=True)
def NormalizeShells(nx, ny, nz, kXofR, kYofR, kZofR, inc, retofROutR, retofROutI, n1ofROut, n2ofROut, NumAtEachR, RMax):
    ResultR = retofROutR.copy()
    ResultI = retofROutI.copy()

    nx2 = nx // 2
    ny2 = ny // 2
    nz2 = nz // 2

    nxOut = nx - 1
    nyOut = ny - 1
    nzOut = nz - 1

    if nx % 2:
        nxOut += 1  # if nx was odd, nxOut=nx
    if ny % 2:
        nyOut += 1  # if ny was odd, nyOut=ny
    if nz % 2:
        nzOut += 1  # if nz was odd, nzOut=nz

    ResEMR = np.zeros((nxOut, nyOut, nzOut), dtype=np.float64)
    ResEMI = np.zeros((nxOut, nyOut, nzOut), dtype=np.float64)
    ResNum = np.zeros((nxOut, nyOut, nzOut), dtype=np.float64)
    ResDen = np.zeros((nxOut, nyOut, nzOut), dtype=np.float64)

    ResEMR[nx2 - 1, ny2 - 1, nz2 - 1] = 1.0
    ResNum[nx2 - 1, ny2 - 1, nz2 - 1] = retofROutR[0][0]
    ResDen[nx2 - 1, ny2 - 1, nz2 - 1] = np.sqrt(n1ofROut[0][0] * n2ofROut[0][0])

    ShapeRR = ResultR.shape
    ShapeRI = ResultI.shape
    ShapeRetRR = retofROutR.shape
    ShapeRetRI = retofROutI.shape
    ShapeN1R = n1ofROut.shape
    ShapeN2R = n2ofROut.shape

    print(ShapeRI, ShapeRR, ShapeRetRR, ShapeRetRI, ShapeN1R, ShapeN2R, inc)
    print('Hello')

    for r in range(1, min(inc + 1, RMax)):
        LastInd = NumAtEachR[r] - 1
        if r % 5 == 1:
            print(r, LastInd)
        retNowR = retofROutR[r][:LastInd]
        retNowI = retofROutI[r][:LastInd]  # Vectors for
        n1Now = n1ofROut[r][:LastInd]
        n2Now = n2ofROut[r][:LastInd]  # given radius
        Num2Surf = LastInd
        for jSurf in range(Num2Surf + 1):  # Fixed the problem!
            retNowRL = retNowR[jSurf]
            retNowIL = retNowI[jSurf]
            n1NowL = n1Now[jSurf]
            n2NowL = n2Now[jSurf]  # Single Values
            if n1NowL * n2NowL == 0:
                continue
            ResultR[r][jSurf] = float(retNowRL / (np.sqrt(n1NowL * n2NowL)))
            ResultI[r][jSurf] = float(retNowIL / (np.sqrt(n1NowL * n2NowL)))
            retNowRL = retNowR[jSurf]
            retNowIL = retNowI[jSurf]
            n1NowL = n1Now[jSurf]
            n2NowL = n2Now[jSurf]  # Single Values
            ResultR[r][jSurf] = float(retNowRL / (np.sqrt(n1NowL * n2NowL)))
            ResultI[r][jSurf] = float(retNowIL / (np.sqrt(n1NowL * n2NowL)))
            kX = int(round(kXofR[r][jSurf]))
            kY = kYofR[r][jSurf]
            kZ = kZofR[r][jSurf]
            if kX == nx2 or kY == nx2 or kZ == nx2:
                continue
            if kX > 0:
                ResEMR[kX + nx2 - 1, kY + ny2 - 1, kZ + nz2 - 1] = ResultR[r][jSurf]
                ResEMR[nx2 - 1 - kX, ny2 - 1 - kY, nz2 - 1 - kZ] = ResultR[r][jSurf]
                ResEMI[kX + nx2 - 1, kY + ny2 - 1, kZ + nz2 - 1] = ResultI[r][jSurf]
                ResEMI[nx2 - 1 - kX, ny2 - 1 - kY, nz2 - 1 - kZ] = -ResultI[r][jSurf]
                ResNum[kX + nx2 - 1, kY + ny2 - 1, kZ + nz2 - 1] = retofROutR[r][jSurf]
                ResNum[nx2 - 1 - kX, ny2 - 1 - kY, nz2 - 1 - kZ] = retofROutR[r][jSurf]
                ResDen[kX + nx2 - 1, kY + ny2 - 1, kZ + nz2 - 1] = np.sqrt(n1ofROut[r][jSurf] * n2ofROut[r][jSurf])
                ResDen[nx2 - 1 - kX, ny2 - 1 - kY, nz2 - 1 - kZ] = np.sqrt(n1ofROut[r][jSurf] * n2ofROut[r][jSurf])
            else:  # kx=0
                ResEMR[nx2 - 1, kY + ny2 - 1, kZ + nz2 - 1] = ResultR[r][jSurf]
                ResEMR[nx2 - 1, -kY + ny2 - 1, -kZ + nz2 - 1] = ResultR[r][jSurf]
                ResEMI[nx2 - 1, kY + ny2 - 1, kZ + nz2 - 1] = ResultI[r][jSurf]
                ResEMI[nx2 - 1, -kY + ny2 - 1, -kZ + nz2 - 1] = ResultI[r][jSurf]
                ResNum[nx2 - 1, kY + ny2 - 1, kZ + nz2 - 1] = retofROutR[r][jSurf]
                ResNum[nx2 - 1, -kY + ny2 - 1, -kZ + nz2 - 1] = retofROutR[r][jSurf]
                ResDen[nx2 - 1, kY + ny2 - 1, kZ + nz2 - 1] = np.sqrt(n1ofROut[r][jSurf] * n2ofROut[r][jSurf])
                ResDen[nx2 - 1, -kY + ny2 - 1, -kZ + nz2 - 1] = np.sqrt(n1ofROut[r][jSurf] * n2ofROut[r][jSurf])

    return ResEMR, ResEMI, ResNum, ResDen, ResultR, ResultI


def main(fNHalfMap1, fNHalfMap2, OutputStringLabel, APixels, dthetaInDegrees, gpu=False):
    if gpu:
        print("Using GPU")

    blockPrint()
    ResultsDir = 'Results_' + OutputStringLabel + '/'
    isResultsDir = os.path.isdir(ResultsDir)
    if 1 - isResultsDir:
        os.mkdir(ResultsDir)

    ResEMOutHDF_FN = ResultsDir + 'ResEM' + OutputStringLabel + 'Out.mrc'
    dthetaInRadians = dthetaInDegrees * np.pi / 180.0
    Thresh = np.cos(dthetaInRadians)
    FTOut = ResultsDir + 'FTPlot' + OutputStringLabel
    PlotsOut = ResultsDir + 'Plots' + OutputStringLabel
    fNResRoot = ResEMOutHDF_FN[0:-4]
    resultAveOut = fNResRoot + 'globalFSC.csv'

    # Section 1       Read in Data, Take Transpose

    h5f_HalfMap1 = mrcfile.open(fNHalfMap1)
    f = h5f_HalfMap1.data

    h5f_HalfMap2 = mrcfile.open(fNHalfMap2)
    g = h5f_HalfMap2.data

    h5f_HalfMap1.close()
    h5f_HalfMap2.close()

    startTime = time.time()

    fT = f.T  # Now it is like EMAN
    gT = g.T

    [nx, ny, nz] = fT.shape

    deltaTime = time.time() - startTime
    print("Maps read in %f seconds for size nx=%g " % (deltaTime, nx))

    # Section 2      Take Fourier Transform; find Cos phase residual

    startTime = time.time()

    fp, gp = ZeroPad(nx, ny, nz, fT, gT)

    deltaTime = time.time() - startTime
    print("NormPad created in %f seconds for size nx=%g " % (deltaTime, nx))

    startTime = time.time()

    F = np.fft.fftn(fp)
    G = np.fft.fftn(gp)
    H = F * np.conj(G)
    HAngle = np.angle(H)
    CosHangle = np.cos(HAngle)

    deltaTime = time.time() - startTime
    print("FFTs performed in %f seconds for size nx=%g " % (deltaTime, nx))

    # Section 3 Create Real Arrays as in original EMAN program

    startTime = time.time()

    d1 = FFTArray2Real(nx, ny, nz, F)
    d2 = FFTArray2Real(nx, ny, nz, G)
    dcH = FFTArray2Real(nx, ny, nz, CosHangle)
    dFPower = FFTArray2Real(nx, ny, nz, F * np.conj(F))

    deltaTime = time.time() - startTime
    print("FFTArray2Real performed in %f seconds for size nx=%g " % (deltaTime, nx))

    # Section 4a Create FSC Outputs; n1 and n2 are the normalizations of
    # f and g. cH means the cosine of the phase residual.

    nx2 = nx // 2
    ny2 = ny // 2
    nz2 = nz // 2

    lsd2 = nx + 2

    dx2 = 1.0 / float(nx2) / float(nx2)
    dy2 = 1.0 / float(ny2) / float(ny2)
    dz2 = 1.0 / float(nz2) / float(nz2)

    inc = max(nx2, ny2, nz2)
    inc = int(inc)

    startTime = time.time()

    retcHGlobal, lr = CreateFTLikeOutputs(inc, nx, ny, nz, dcH, nx2, ny2, nz2, dx2, dy2, dz2)
    ret, n1, n2, lr = CreateFSCOutputs(inc, nx, ny, nz, d1, d2, nx2, ny2, nz2, dx2, dy2, dz2)
    FPower, lr = CreateFTLikeOutputs(inc, nx, ny, nz, dFPower, nx2, ny2, nz2, dx2, dy2, dz2)

    deltaTime = time.time() - startTime
    print("CreateFSCOutputs performed in %f seconds for size nx=%g " % (deltaTime, nx))

    # Section 4b      Write out FSCs. Define RMax based on this

    linc = 0
    for i in range(inc + 1):
        if lr[i] > 0:
            linc += 1

    result = [0 for i in range(3 * linc)]

    ii = -1
    for i in range(inc + 1):
        if lr[i] > 0:
            ii += 1
            result[ii] = float(i) / float(2 * inc)
            result[ii + linc] = float(ret[i] / (np.sqrt(n1[i] * n2[i])))
            result[ii + 2 * linc] = lr[i]  # Number of Values

    NormalizedFreq = result[0:(inc + 1)]
    resultAve = result[(inc + 1):(2 * (inc + 1))]  # This takes values inc+1 values from inc+1 to 2*inc+1

    with open(resultAveOut, "w") as fL1:
        AveWriter = csv.writer(fL1)
        for j in range(inc + 1):
            valFreqNormalized = NormalizedFreq[j]
            valFreq = valFreqNormalized / APixels
            valFSCshell = resultAve[j]
            AveWriter.writerow([valFreqNormalized, valFreq, valFSCshell])

    aa = np.abs(np.array(resultAve)) < .13
    bb = np.where(aa)[0]
    try:
        RMax = bb[0] + 4
    except:
        RMax = inc

    RMax = min(RMax, inc)
    print('simple FSC written out to ' + resultAveOut)
    print('RMax = %d' % RMax)

    Nxf = np.int32(nx2) + 1
    k0 = np.array(range(Nxf)) / APixels / 2.0 / Nxf
    fig, ax = plt.subplots()
    ax.plot(k0, np.log(FPower), 'b', label='FPower')
    ax.set_xlabel('Spatial Frequency (1/A) ')
    ax.set_ylabel('log rot ave FT Power')

    fig.savefig(FTOut + '.jpg')
    fig, ax = plt.subplots()
    ax.plot(k0, 2 * retcHGlobal / lr, 'b', label='ave cos phase')
    ax.plot(k0, resultAve, 'g', label='FSC')
    ax.set_xlabel('Spatial Frequency (1/A) ')
    ax.set_ylabel('various FSCs')

    # Section 5. Create generalized FSC  and FT arrays
    startTime = time.time()

    kXofR, kYofR, kZofR, retofRR, retofRI, n1ofR, n2ofR, NumAtEachR = \
        createFSCarrays(nx, ny, nz, lsd2, lr, inc, dx2, dy2, dz2, d1, d2, nx2, ny2, nz2)

    kXofR, kYofR, kZofR, retcH, retFT, n12ofR = \
        createFTarrays(nx, ny, nz, lsd2, lr, inc, dx2, dy2, dz2, dcH, dFPower, nx2, ny2, nz2)

    deltaTime = time.time() - startTime

    print("FSC arrays created in %f seconds for size nx=%g " % (deltaTime, nx))

    NumAtEachRMax = NumAtEachR[-1]

    kXofR = kXofR[:, :NumAtEachRMax]
    kYofR = kYofR[:, :NumAtEachRMax]
    kZofR = kZofR[:, :NumAtEachRMax]
    retofRR = retofRR[:, :NumAtEachRMax]
    retofRI = retofRI[:, :NumAtEachRMax]
    n1ofR = n1ofR[:, :NumAtEachRMax]
    n2ofR = n2ofR[:, :NumAtEachRMax]

    # Section 6. Average on Shells

    startTime = time.time()

    if gpu:
        retofROutR, retofROutI, n1ofROut, n2ofROut, NumAtROut = \
            cuda_functions.AveragesOnShellsUsingLogicBCuda(inc, retofRR, retofRI, n1ofR, n2ofR, kXofR, kYofR, kZofR,
                                                           NumAtEachR, Thresh, RMax)
    else:

        retofROutR, retofROutI, n1ofROut, n2ofROut, NumAtROut = \
            AveragesOnShellsUsingLogicB(inc, retofRR, retofRI, n1ofR, n2ofR, kXofR, kYofR, kZofR, NumAtEachR, Thresh,
                                        RMax)

    deltaTime = time.time() - startTime

    print("AveragesOnShells created in %f seconds for size nx=%g " % (deltaTime, nx))

    # Section 7. We have the unaveraged quantities

    startTime = time.time()

    ResEMR, ResEMI, ResNum, ResDen, ResultR, ResultI = \
        NormalizeShells(nx, ny, nz, kXofR, kYofR, kZofR, inc, retofROutR, retofROutI, n1ofROut, n2ofROut, NumAtEachR,
                        RMax)
    deltaTime = time.time() - startTime

    print("NormalizeShells created in %f seconds for size nx=%g, RMax=%g " % (deltaTime, nx, RMax))

    ResEMRT = ResEMR.T
    mrc_write = mrcfile.new(ResEMOutHDF_FN, overwrite=True)
    mrc_write.set_data(ResEMRT.astype('<f4'))
    mrc_write.voxel_size = (float(APixels), float(APixels), float(APixels))
    mrc_write.update_header_from_data()
    mrc_write.close()

    # Section 10 Plot 5 Axes

    xf, yf, zf = ExtractAxes(ResEMR)
    Nxf = len(xf)
    DPRf = 2 * retcHGlobal[:-1] / lr[:-1]
    Globalf = resultAve[:-1]
    pltName = PlotsOut
    k0 = np.array(range(Nxf)) / APixels / 2.0 / Nxf

    fig, ax = plt.subplots()
    ax.plot(k0, xf, 'b', label='x dir')
    ax.plot(k0, yf, 'g', label='y dir')
    ax.plot(k0, zf, 'r', label='z dir')
    ax.plot(k0, DPRf, 'k', label='ave cos phase')
    ax.plot(k0, Globalf, 'y', label='global FSC')

    ax.set_xlabel('Spatial Frequency (1/A) ')
    ax.set_ylabel('FSCs')

    # Now add the legend with some customizations.
    fig.savefig(pltName + '.jpg')
    xyzf = np.reshape(np.concatenate((xf, yf, zf, DPRf, Globalf)), (5, Nxf))
    xyzf = xyzf.T
    np.savetxt(PlotsOut + '.csv', xyzf)

    ## Flush out plots
    plt.clf()
    plt.cla()
    plt.close()

    enablePrint()


if __name__ == "__main__":
    main(argv[1], argv[2], argv[3], float(argv[4]), float(argv[5]), bool(argv[6]))
