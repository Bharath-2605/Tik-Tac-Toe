import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = HandDetector(detectionCon=0.8,maxHands=2)
class TicTacToe:
    def __init__(self):
        self.board=np.zeros((3,3),dtype=int)
        self.gameOver=False
        self.winner=None
        self.currentPlayer=1  
        self.cellSize=100
        self.offsetX=500
        self.offsetY=200
    def drawBoard(self,img):
        for x in range(1,3):
            cv2.line(img,(self.offsetX+x*self.cellSize,self.offsetY),
                     (self.offsetX+x*self.cellSize,self.offsetY+3*self.cellSize),(255,255,255),3)
            cv2.line(img,(self.offsetX,self.offsetY+x*self.cellSize),
                     (self.offsetX+3*self.cellSize,self.offsetY+x*self.cellSize),(255,255,255),3)
        for y in range(3):
            for x in range(3):
                cellValue=self.board[y, x]
                centerX=self.offsetX+x*self.cellSize+self.cellSize//2
                centerY=self.offsetY+y*self.cellSize+self.cellSize//2
                if cellValue==1:  
                    cv2.line(img,(centerX-20,centerY-20),(centerX+ 20, centerY + 20), (0, 0, 255), 3)
                    cv2.line(img, (centerX + 20, centerY - 20), (centerX - 20, centerY + 20), (0, 0, 255), 3)
                elif cellValue == -1: 
                    cv2.circle(img, (centerX, centerY), 30, (255, 0, 0), 3)
    def makeMove(self, x, y):
        if self.board[y, x] == 0:
            self.board[y, x] = self.currentPlayer
            self.currentPlayer *= -1  
            self.checkGameOver()
    def checkGameOver(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                self.gameOver = True
                self.winner = 1 if sum(self.board[i, :]) > 0 else -1
                return
            if abs(sum(self.board[:, i])) == 3:
                self.gameOver = True
                self.winner = 1 if sum(self.board[:, i]) > 0 else -1
                return
        if abs(sum(self.board.diagonal())) == 3:
            self.gameOver = True
            self.winner = 1 if sum(self.board.diagonal()) > 0 else -1
            return
        if abs(sum(np.fliplr(self.board).diagonal()))== 3:
            self.gameOver=True
            self.winner=1 if sum(np.fliplr(self.board).diagonal()) > 0 else -1
            return
        if not self.gameOver and np.all(self.board != 0):
            self.gameOver=True
            self.winner=0  
    def getCell(self, px, py):
        if self.offsetX < px < self.offsetX + 3 * self.cellSize and self.offsetY < py < self.offsetY + 3 * self.cellSize:
            cellX=(px - self.offsetX)//self.cellSize
            cellY=(py - self.offsetY)//self.cellSize
            return cellX, cellY
        return None
game = TicTacToe()
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)
    if hands and not game.gameOver:
        for hand in hands:
            lmList = hand['lmList']  
            pointIndex = lmList[8][0:2]  
            cell = game.getCell(*pointIndex)
            handType = hand['type']
            if cell:
                if game.currentPlayer==1 and handType=="Right":
                    game.makeMove(cell[0], cell[1])
                elif game.currentPlayer == -1 and handType=="Left":
                    game.makeMove(cell[0], cell[1])
    game.drawBoard(img)
    if game.gameOver:
        if game.winner == 1:
            cvzone.putTextRect(img, "Player 1 Wins!", [500, 100], scale=3, thickness=2, offset=10, colorR=(0, 200, 0))
        elif game.winner == -1:
            cvzone.putTextRect(img, "Player 2 Wins!", [500, 100], scale=3, thickness=2, offset=10, colorR=(200, 0, 0))
        elif game.winner == 0:
            cvzone.putTextRect(img, "It's a Draw!", [500, 100], scale=3, thickness=2, offset=10, colorR=(255, 255, 0))
    cv2.imshow("Tic Tac Toe", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game = TicTacToe()  
    elif key == ord('e'):
        break
cap.release()
cv2.destroyAllWindows()
