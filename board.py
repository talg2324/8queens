import numpy as np

class Solver():
    def __init__(self, n) -> None:
        self.board = np.zeros((n,n), dtype=np.uint8)
        self.n = n

        j, i = np.meshgrid(np.arange(0, n), np.arange(0, n))
        self.updiag = i + j
        self.downdiag = i - j
        self.nqueens = 0
        self.is_solved = False

        self.i = self.j = 0

    def solve(self):
        while not self.is_solved:
            self.single_iter()

    def next_solution(self):
        self.is_solved = False
        self.i, self.j = self.go_back(self.i)

    def single_iter(self):
        if self.nqueens == self.n:
            self.is_solved = True

        # Place a queen
        if self.valid_move(self.i, self.j):
            self.board[self.i, self.j] = 1
            self.nqueens += 1
            self.i += 1
            self.j = 0

        # Go back a row
        elif self.j == self.n - 1:
            self.i, self.j = self.go_back(self.i)

        else:
            self.j += 1

        return self.board

    def go_back(self, i):
            j = int(np.where(self.board[i-1, :])[0])
            self.nqueens -= 1
            self.board[i-1, j] = 0
            
            if j == self.n - 1:
                return self.go_back(i-1)
            else:
                return (i-1, j+1)

    def valid_move(self, i, j):

        updiag = i+j
        downdiag = i-j
        
        if np.any(self.board[:, j]):
            ret = False
        
        elif np.any(self.board[i, :]):
            ret = False

        elif np.any(self.board[self.updiag==updiag]):
            ret = False
        
        elif np.any(self.board[self.downdiag==downdiag]):
            ret = False
        
        else:
            ret = True

        return ret
    



    def __repr__(self) -> str:
        """
        Print a chess board using regular keys
        """

        outstr = ''

        for i in range(self.n):
            outstr += " ___"
        outstr += "\n"

        for i in range(self.n):
            outstr += "|"
            for j in range(self.n):
                if self.board[i,j]:
                    tile = 'Q'
                elif (i+j)%2==0:
                    tile = '#'
                else:
                    tile = '_'

                outstr += "_%s_|"%tile
            outstr += "\n"
        return outstr

if __name__ == "__main__":

    b = Solver(8)

    # Print 10 solutions
    for i in range(10):
        b.solve()
        print(b)
        b.next_solution()

    print('Done')