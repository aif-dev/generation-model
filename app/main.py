import pygame
import components

def LoadFileIntoMusicPlayer(listViewer, musicPlayer):
    file = listViewer.GetSelectedFile()
    if file:
        musicPlayer.LoadFile(file.file)

def main():
    #Initializes Pygame Application
    pygame.init()
    #Setting apps dimensions
    WIDTH = 800
    HEIGHT = 600
    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    #Setting apps caption
    pygame.display.set_caption("JukeBox AI")
    #Creating title
    font = pygame.font.SysFont(None, 40)
    title = font.render("Welcome to JukeBox AI!", True, (255,255,255))
    #Title box
    title_rect = title.get_rect()
    title_rect.center = (WIDTH // 2, 40)
    #Initialize generate Button
    genButton = components.GenerateButton(50, 100, 100, 30)
    genButton.SetText("Generate Music")
    #Initialize ListViewer
    listViewer = components.ListViewer(WIDTH//2, 100, (WIDTH//2) - 10, HEIGHT//2)
    mockList = ["Song 1", "Song 2","Song 3"]
    listViewer.StoreList(mockList)
    #Initialize Load Button
    loadButton = components.LoadButton(WIDTH//2, (WIDTH//2 + 7), 100, 30)
    loadButton.SetText("Load File")
    #Initialize Music player
    musicPlayer = components.MusicPlayer(0,HEIGHT-150,WIDTH,150)
    #Initialize Select button
    selectButton = components.SelectButton(loadButton.x + loadButton.width + 10, (WIDTH//2 + 7), 100, 30)
    selectButton.SetText("Select File")
    #Initialize refresh list button
    refreshButton = components.RefreshListButton(selectButton.x + selectButton.width + 10, (WIDTH//2 + 7), 100, 30)
    refreshButton.SetText("Refresh List")

    running = True
    while running:

        screen.fill((32,32,32))
        screen.blit(title, title_rect)
        mouse = pygame.mouse.get_pos()
        genButton.AddToScreen(screen, mouse)
        listViewer.AddToScreen(screen, mouse)
        loadButton.AddToScreen(screen, mouse)
        musicPlayer.AddToScreen(screen, mouse)
        selectButton.AddToScreen(screen, mouse)
        refreshButton.AddToScreen(screen, mouse)

        for e in pygame.event.get():
            #Event for shutting own application
            if e.type == pygame.QUIT:
                pygame.quit()

            if e.type == pygame.MOUSEBUTTONDOWN:
                genButton.MouseDownHandler(mouse)
                listViewer.MouseDownHandler(mouse)
                if loadButton.MouseDownHandler(mouse) == True:
                    LoadFileIntoMusicPlayer(listViewer, musicPlayer)
                selectButton.MouseDownHandler(mouse)
                musicPlayer.MouseDownHandler(mouse)
                refreshButton.MouseDownHandler(mouse)

        pygame.display.flip()

main()
