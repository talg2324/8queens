from board import Solver
import cv2
import numpy as np

N_QUEENS = 8

def main():
    board_im, H = calibrate()
    queen, alpha = load_queen()

    board_solver = Solver(N_QUEENS)
    writer = cv2.VideoWriter("./samples/solution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 100, (board_im.shape[1], board_im.shape[0]), True)

    while not board_solver.is_solved:
        solution, steps = run_solution_finder(writer, board_solver, board_im, queen, alpha, H)

        cv2.putText(solution, "Press space bar for next solution or any other key to exit", (5,50), cv2.FONT_ITALIC, 1, (255,255,255))
        cv2.putText(solution, "Solved in %d steps"%steps, (5,80), cv2.FONT_ITALIC, 1, (255,255,255))
        cv2.imshow('Board Position', solution)
        c = cv2.waitKey(0)

        for i in range(100):
            writer.write(solution)

        if c == 32: # Press space bar to continue
            board_solver.next_solution()

    writer.release()
            
def run_solution_finder(writer, board_solver, board_im, queen, alpha, H):
    steps = 0
    while not board_solver.is_solved:
        board = board_solver.single_iter()
        im = update_gui(board, board_im, queen, alpha, H)
        writer.write(im)
        steps += 1
    return im, steps


def update_gui(board, board_im, queen, alpha, H):
        i, j = np.where(board==1)
        board_im_t = board_im.copy()

        # Place each queen that is already on the board
        for k in range(i.shape[0]):
            place_at(board_im_t, queen, alpha, H, (j[k]/N_QUEENS, i[k]/N_QUEENS))
            
        cv2.imshow('Board Position', board_im_t)
        cv2.waitKey(1)
        return board_im_t

def place_at(board_im, queen, alpha, H, point):
    """
    Find the queen's coordinates in (y,x)
    Map to (u,v) using the Homography matrix
    Place the queen
    """
    p1 = cv2.perspectiveTransform(np.array(point).reshape(1,1,2), H)
    px, py = np.squeeze(p1).astype(np.int32)

    w = queen.shape[1]//2
    h = queen.shape[0]//2

    board_im[py-2*h+5:py+6, px-w:px+w+1, :] = queen * alpha + (1-alpha) * board_im[py-2*h+5:py+6, px-w:px+w+1, :]


def load_queen():
    queen_im = cv2.imread('./images/queen_matted.jpg')
    alpha = np.load('./images/queen_alphamap.npy')
    bby, bbx = alpha.nonzero()
    queen_im = queen_im[bby.min():bby.max(), bbx.min():bbx.max(), :]
    alpha = alpha[bby.min():bby.max(), bbx.min():bbx.max()]
    return queen_im, cv2.cvtColor(alpha.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def calibrate():
    """
    1) Find the 7x7 inner corners of the chess board (v, u)
    2) Create a meshgrid representing the (y, x) coordinates in the interval [0, 1]
    3) Find the homography from (y, x) to (v, u)
    4) Use the alpha map to add a custom background to the chess board
    """
    board_im = np.pad(cv2.imread('./images/board_matted.jpg'), ((20,20), (20,20), (0,0)))
    board_alpha = np.pad(np.load('./images/board_alphamap.npy'), ((20,20), (20,20)))
    board_alpha = cv2.cvtColor(board_alpha.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    background = cv2.resize(cv2.imread('./images/background.jpg'), (board_im.shape[1], board_im.shape[0]), cv2.INTER_CUBIC)

    ret, corners = cv2.findChessboardCornersSB(board_im, (7, 7), flags=cv2.CALIB_CB_EXHAUSTIVE)

    if ret:
        # Set the needed parameters to find the refined corners
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
        corners = cv2.cornerSubPix(cv2.cvtColor(board_im, cv2.COLOR_BGR2GRAY), corners, winSize, zeroZone, criteria)
        corners = np.squeeze(corners)
        

        # Match points from (v, u) to (y, x)
        scale = 50
        r = scale//2
        board_x, board_y = np.meshgrid(np.linspace(0, 1, scale * N_QUEENS), np.linspace(0, 1, scale * N_QUEENS))
        matching_points = []
        for i in range(N_QUEENS-1):
            for j in range(N_QUEENS-1):
                p = (board_x[scale*i+r, scale*j+r], board_y[scale*i+r, scale*j+r])
                matching_points.append(p)
        
        matching_points_src = np.expand_dims(np.stack(matching_points), 1)
        matching_points_dst = np.expand_dims(corners, 1)
        H, _ = cv2.findHomography(matching_points_src, matching_points_dst)
        board_im = board_alpha * board_im + (1-board_alpha) * background
        return board_im, H


if __name__ == "__main__":
    main()