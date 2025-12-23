### 3DFSC Program Suite Version 3.1 (5 October 2023) ###

This is the most recent version of 3DFSC, by Philip Baldwin, Yong Zi Tan, and Dmitry Lyumkis.

GPU code and Conda environment by Carl Negro.

3DFSC Program Suite requires Miniconda 3 to run, and UCSF Chimera or ChimeraX to visualize the outputs.

**Important: this version of the software has been modified to work with updated dependencies (October 2023). I have tested it with cudatoolkit 11.8**

## Installation ##

1) ``conda create -y -n fsc3D-3.0 "python<3.12" cudatoolkit=11.8``
2) ``conda activate fsc3D-3.0``
3) ``pip install numpy numba==0.58.0 scipy click h5py scikit-image matplotlib mrcfile``
4) ``git clone https://github.com/azazellochg/fsc3D fsc3D-3.0``

## Execution ##

To view the 3DFSC parameters, access the help info like `ThreeDFSC/ThreeDFSC_Start.py -h`.

**Important notes:**
- The `--apix` parameter is **required** and must be set to the correct pixel size of your map. The default value of 1.0 is incorrect and should not be used.
- Use `--viewer` to specify the visualization tool: `chimera` (default) or `chimerax`.

**Example 1: Using Chimera (default viewer)**
```bash
python ThreeDFSC/ThreeDFSC_Start.py \
--halfmap1 <path_to_halfmap1.mrc> \
--halfmap2 <path_to_halfmap2.mrc> \
--fullmap <path_to_fullmap.mrc> \
--mask <path_to_mask.mrc> \
--ThreeDFSC <output_directory> \
--gpu \
--apix <pixel_size>
```

**Example 2: Using ChimeraX**
```bash
python ThreeDFSC/ThreeDFSC_Start.py \
--halfmap1 <path_to_halfmap1.mrc> \
--halfmap2 <path_to_halfmap2.mrc> \
--fullmap <path_to_fullmap.mrc> \
--mask <path_to_mask.mrc> \
--ThreeDFSC <output_directory> \
--gpu \
--apix <pixel_size> \
--viewer chimerax
```

## GPU Execution ##

3DFSC now has GPU support through Numba for faster execution (typically ~10x faster than the CPU implementation). This requires that CUDA is installed correctly. See http://www.nvidia.com/Download/index.aspx. 

To make use of a GPU, simply append the `--gpu` flag as a parameter to ThreeDFSC_Start.py script.

You can select which GPU to use for processing with the `--gpu_id` flag. E.g. `--gpu_id=2` to run 3DFSC on the GPU with index number 2. The program currently only allows a single GPU to be used at once; this may change in future versions.

To see a list of available GPU's and corresponding indices, run `nvidia-smi`. If you are unable to run `nvidia-smi`, check to make sure you have CUDA installed correctly.

Note that GPU memory is limited, so that processing jobs with large box sizes will fail. 

Questions and feedback welcomed, and should be sent to prbprb2@gmail.com, ytan@nysbc.org and dlyumkis@salk.edu.
