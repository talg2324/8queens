# 8queens
Utilize computer vision techniques to visualize the solution to the classic bruteforcing problem
<p align="center" width="100%">
    <img width="50%" src="./samples/solution.gif"> 
</p>

N-Queens Solver:
- Solve N-Queens by bruteforce
- Continue from the last solution to find the next one

Visualization:
- Background/Foreground segmentation with scribbles to separate the chess set from the background
- Use the Djikstra algorithm to propagate scribble information into a trimap
- Trimap Refinement
- Create a homography mapping the "digital" queen placement onto the chessboard

1) Trimap Generation


<p align='center'>Input Image -> Foreground Probability Map -> Foreground Distance Map -> Final Trimap (Foreground + Background)</p>
<p align="center" width="100%">
    <img width="50%" src="./samples/queen_matting.png"> 
</p>

<p align="center" width="100%">
    <img width="50%" src="./samples/board_matting.png"> 
</p>

2) Trimap Refinement
<p align="center" width="100%">
    <img width="50%" src="./samples/queen_alphamap.png"> 
</p>

<p align="center" width="100%">
    <img width="50%" src="./samples/board_alphamap.png"> 
</p>

3) Bruteforce!
