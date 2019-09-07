from MenuItem import MenuItem
#This class represents a menu object.
class Menu(object):
    MENU_ITEM_TYPES = ["Drink","Appetizer","Entree","Dessert"]
    def __init__(self,menuFile):
        self.menuItemDictionary = {}
        fileIn = open(menuFile,"r")
        for line in fileIn:
            line = line.strip()
            line = line.split(",")
            if line[1] not in self.menuItemDictionary:
                self.menuItemDictionary[line[1]] = [MenuItem(line[0],line[1],line[2],line[3])]
            elif line[1] in self.menuItemDictionary:
                self.menuItemDictionary[line[1]].append(MenuItem(line[0],line[1],line[2],line[3]))
        fileIn.close()
    #fn: getMenuItem
    #input: foodType(str), index(int)
    #return: MenuItem Object
    def getMenuItem(self,foodType,index):
        return self.menuItemDictionary[foodType][index]
    #fn: printMenuItemsByType
    #input: foodType (str)
    #side-effect: Prints Menu items in certain type
    #return: none
    def printMenuItemsByType(self,foodType):
        counter = 0
        print("-----"+foodType+"-----")
        for food in self.menuItemDictionary[foodType]:
            print(str(counter)+")",food)
            counter += 1
    #fn: getNumMenuItemsByType
    #input: foodType (str)
    #return: Length of item list (int)
    def getNumMenuItemsByType(self,foodType):
        return len(self.menuItemDictionary[foodType])

