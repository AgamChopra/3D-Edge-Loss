<h1>3D Edge Loss</h1>
<p>PyTorch code to implement a gradient magnitude based edge detection loss.</p>

<p align="center">
    <img width="300" height="300" src="https://github.com/AgamChopra/3D-Edge-Loss/blob/main/imgs/grad_mag.gif">   
    <img width="300" height="300" src="https://github.com/AgamChopra/3D-Edge-Loss/blob/main/imgs/edges_fullsob_dev.gif">  
    <br><i>Fig. Map of the magnitude of the gradients(Left-old). Map of the magnitude of the gradients with diagonal edges(Right-new/dev).</i><br>  
    <img width="300" height="300" src="https://github.com/AgamChopra/3D-Edge-Loss/blob/main/imgs/Figure%202022-07-12%20141758%20(42).png">
    <img width="300" height="300"src="https://github.com/AgamChopra/3D-Edge-Loss/blob/main/imgs/Figure%202022-07-12%20141758%20(43).png">
    <br><i>Fig. Input after 3D gaussian blur(Left). Gradient wrt x(Right)</i><br>   
    <img width="300" height="300" src="https://github.com/AgamChopra/3D-Edge-Loss/blob/main/imgs/Figure%202022-07-12%20141758%20(44).png">
    <img width="300" height="300"src="https://github.com/AgamChopra/3D-Edge-Loss/blob/main/imgs/Figure%202022-07-12%20141758%20(45).png">
    <br><i>Fig. Gradient wrt y(Left). Gradient wrt z(Right).</i><br>   
    <img width="300" height="300" src="https://github.com/AgamChopra/3D-Edge-Loss/blob/main/imgs/Figure%202022-07-12%20141758%20(46).png">
    <img width="300" height="300"src="https://github.com/AgamChopra/3D-Edge-Loss/blob/main/imgs/Figure%202022-07-12%20141758%20(47).png">
    <br><i>Fig. Magnitude of gradients(Left). Filtered magnitude of gradients.(Right).</i><br>
</p>

<p><a href="https://raw.githubusercontent.com/AgamChopra/WGAN-GP/main/LICENSE" target="blank">[GNU AGPL3 License]</a></p>
