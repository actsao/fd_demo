# Fractal Dimension Box Counting (Windows Demo)

![Measuring the coast of Britain](https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Great_Britain_Box.svg/320px-Great_Britain_Box.svg.png)

Slightly modified to work in Windows.

## Installation
1.) Clone or download the repository using https://github.com/actsao/fd_demo.git for cloning or under Code > ZIP for downloading. Extract files if needed.
 - Download recommended for new github users

2.) In CMD, run

```
conda env create -f environment.yml
```

## Sample Usage
1.) In Anaconda Prompt, activate the environment

```
conda activate fd_demo
```

- Anaconda Prompt is recommended over CMD for ease of use. Either is fine otherwise.

2.) CD into the folder containing the downloaded files

3.) Run the following sample command

```
python fractalDim_graph_no_threshold.py -f 013_pect_phfirst_070511
```

## Additional notes
- Code presented here is largely the same as the file with the same name on the cluster with slight modifications to work on Windows
   - Stencil function uses Window version called "GenerateStenciledLabelMapFromParticles.exe"
      - Stencil function copied from "SlicerCIP4-10-2-win" and has a last modified date of 06/28/2019
      - Stencil function and requisite files are under the "lib" folder
   - Automatic filename ending detector does not work (cannot automatically differentiate between left/right or total vtk particle files)
- Some functions might not work. I haven't thoroughly tested for Windows compatibility.
