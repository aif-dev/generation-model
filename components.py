import pygame
import tkinter
import tkinter.filedialog

GR55 = (55, 55, 55)
GR90 = (90, 90, 90)
GR130 = (130, 130, 130)
GR170 = (170, 170, 170)


class Button:
    def __init__(self, x, y, width, height, color=GR90):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def SetText(self, text, lsize=24, color=(255, 255, 255)):
        font = pygame.font.SysFont("Corbel", int(lsize))
        self.text = font.render(text, True, color)
        self.width = self.text.get_rect().width + 40
        self.height = self.text.get_rect().height + 20

    def AddText(self, screen):
        screen.blit(self.text, (self.x + 20, self.y + 10))

    def AddToScreen(self, screen, mouse):
        self.OnHoverHandler(screen, mouse)
        self.AddText(screen)

    def OnHoverHandler(self, screen, mouse):
        if self.x <= mouse[0] <= (self.x + self.width) and self.y <= mouse[1] <= (
            self.y + self.height
        ):
            pygame.draw.rect(screen, GR130, [self.x, self.y, self.width, self.height])
        else:
            pygame.draw.rect(
                screen, self.color, [self.x, self.y, self.width, self.height]
            )

    def MouseDownHandler(self, mouse):
        pass


class GenerateButton(Button):
    def MouseDownHandler(self, mouse):
        if self.x <= mouse[0] <= (self.x + self.width) and self.y <= mouse[1] <= (
            self.y + self.height
        ):
            return True
        else:
            return False


class SelectButton(Button):
    def MouseDownHandler(self, mouse):
        if self.x <= mouse[0] <= (self.x + self.width) and self.y <= mouse[1] <= (
            self.y + self.height
        ):
            return True
        else:
            return False


class LoadButton(Button):
    def MouseDownHandler(self, mouse):
        if self.x <= mouse[0] <= (self.x + self.width) and self.y <= mouse[1] <= (
            self.y + self.height
        ):
            top = tkinter.Tk()
            top.withdraw()  # hide window
            file_name = tkinter.filedialog.askopenfilename(parent=top)
            top.destroy()
            return file_name
        else:
            return False


class RefreshListButton(Button):
    def MouseDownHandler(self, mouse):
        if self.x <= mouse[0] <= (self.x + self.width) and self.y <= mouse[1] <= (
            self.y + self.height
        ):
            return True
        else:
            return False


class PlayPauseButton(Button):
    def SetText(self, text, lsize=24, color=(255, 255, 255)):
        font = pygame.font.SysFont("Corbel", int(lsize))
        self.text = font.render(text, True, color)

    def AddText(self, screen):
        screen.blit(
            self.text,
            (
                self.x + (self.width // 2) - (self.text.get_rect().width / 2),
                self.y + (self.height // 2) - (self.text.get_rect().height // 2),
            ),
        )

    def MouseDownHandler(self, mouse):
        if self.x <= mouse[0] <= (self.x + self.width) and self.y <= mouse[1] <= (
            self.y + self.height
        ):
            print("Play/Pause Button")
            print(self.IsPlaying)
            if self.IsPlaying:
                self.SetText("Play")
                self.IsPlaying = False
            else:
                self.SetText("Pause")
                self.IsPlaying = True


class ListViewer:
    def __init__(self, x, y, width, height, color=GR55):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.list = []

    def AddToScreen(self, screen, mouse):
        pygame.draw.rect(
            screen, self.color, [self.x, self.y, self.width, self.height], self.width, 5
        )
        for item in self.list:
            item.AddToScreen(screen, mouse)

    def StoreList(self, list):
        # Initialize list
        yi = self.y + 10
        for item in list:
            listItem = ListItem(self.x + 5, yi, self.width - 10, 30)
            listItem.SetText(item.split("/")[-1])
            listItem.file = item
            self.list.append(listItem)
            yi = yi + 30

    def DeselectAllList(self):
        for item in self.list:
            item.selected = False

    def MouseDownHandler(self, mouse):
        if self.x <= mouse[0] <= (self.x + self.width) and self.y <= mouse[1] <= (
            self.y + self.height
        ):
            self.DeselectAllList()
            for item in self.list:
                item.MouseDownHandler(mouse)

    def GetSelectedFile(self):
        for item in self.list:
            if item.selected == True:
                return item
        return ""


class ListItem:
    def __init__(self, x, y, width, height, color=GR55):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.selected = False

    def SetText(self, text, lsize=24, color=(255, 255, 255)):
        font = pygame.font.SysFont("Corbel", int(lsize))
        self.text = font.render(text, True, color)

    def AddText(self, screen):
        screen.blit(self.text, (self.x + 5, self.y + 5))

    def OnHoverHandler(self, screen, mouse):
        if self.selected == True:
            pygame.draw.rect(screen, GR170, [self.x, self.y, self.width, self.height])
        else:
            if self.x <= mouse[0] <= (self.x + self.width) and self.y <= mouse[1] <= (
                self.y + self.height
            ):
                pygame.draw.rect(
                    screen, GR130, [self.x, self.y, self.width, self.height]
                )
            else:
                pygame.draw.rect(
                    screen, self.color, [self.x, self.y, self.width, self.height]
                )

    def AddToScreen(self, screen, mouse):
        self.OnHoverHandler(screen, mouse)
        self.AddText(screen)

    def MouseDownHandler(self, mouse):
        if self.x <= mouse[0] <= (self.x + self.width) and self.y <= mouse[1] <= (
            self.y + self.height
        ):
            self.selected = True


class MusicPlayer:
    def __init__(self, x, y, width, height, color=GR55):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.controlButton = PlayPauseButton(self.x + 20, self.y + 10, 130, 130)
        self.controlButton.SetText("Play")
        self.controlButton.IsPlaying = False
        self.currentFile = "No file has been loaded"

    def AddToScreen(self, screen, mouse):
        pygame.draw.rect(screen, GR55, [self.x, self.y, self.width, self.height])
        self.SetText(self.currentFile)
        screen.blit(self.text, (self.x + 170, self.y + 20))
        self.controlButton.AddToScreen(screen, mouse)

    def SetText(self, text, lsize=24, color=(255, 255, 255)):
        font = pygame.font.SysFont("Corbel", int(lsize))
        self.text = font.render(text, True, color)

    def LoadFile(self, file):
        self.currentFile = file
        print(self.currentFile)
        self.SetText(self.currentFile)

    def MouseDownHandler(self, mouse):
        self.controlButton.MouseDownHandler(mouse)
