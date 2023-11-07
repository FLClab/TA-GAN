## Server side

1) Download the trained models '5000_net_G.pth' and '5000_net_S.pth' and save them in checkpoints/LiveFActin.
2) Build dockerfile 'taganserver':

`sudo docker build -t taganserver .`

3) Run the docker image
4) 
`sudo docker run --gpus all -it --rm --user $(id -u) -p 5000:5000 --shm-size=10g -v /:/workspace/ taganserver`

`cd ..`

`cd workspace`

6) Run the server function
   
`python server_v2.py`

The server is now ready to receive data and commands from the microscope side.


## Microscope side

1) Activate environment (TODO: add requirements.txt from py37)
2) Run the automatic acquisition script

` python auto_region_select_multiregion.py`

Parameters that can be played with in the script:
- px = 100 (size in pixels of subregions; should be a factor of width and height; minimum is 50 pixels with current version of specpy, smaller regions can't be acquired with STED)
- ROI = 300 (size of central ROI where subregions are not acquired; ROI + 2(px) should be equal to width and height, but this is not inforced)
- width, height = 500.0\*20\*1e-9, 500.0\*20\*1e-9 (field of view in meters of each selected regions; we've used square regions only, rectangular regions would need to be tested)
- repetitions = 15 (number of frames acquired)
- tlim = 60 (time in seconds between each frame)

For controls (acquiring only STED images at every time step), run 
` python auto_region_select_multiregion_control.py`

3) Follow the instructions, which will ask you to:
   3.1 Type the name identifying the coverslip
   3.2 Select the STED configuration window in Imspector, then press enter in the terminal
   3.3 Select the confocal configuration window in Imspector, then press enter in the terminal
   3.4 Select tht overview configuration window in Imspector, then press enter in the terminal
   3.5 Type the name of the overview window (normally '640'), then press enter in the terminal
   3.6 The overview window should open, select a few regions to image (we always chose three regions for the results in the paper)
   3.7 Let the microscope and the server do the whole acquisition for you!
