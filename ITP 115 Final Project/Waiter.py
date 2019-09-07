from Menu import Menu
from Diner import Diner
#This class represents the waiter object.
class Waiter(object):
    def __init__(self,MenuObj):
        self.diners = []
        self.menu = MenuObj
    #fn addDiner
    #input: Diner object
    #return: None
    def addDiner(self,Diner):
        self.diners.append(Diner)
    #fn getNumDiners:
    #input: None
    #return Number diners (int)
    def getNumDiners(self):
        return len(self.diners)
    #fn printDinerStatuses
    #input: None
    #side-effect: Prints statuses and diner in each status
    #return: None
    def printDinerStatuses(self): #Fix later
        for index in range(len(Diner.STATUSES)):
            print("Diners who are currently "+Diner.STATUSES[index]+":")
            if self.getNumDiners() != 0:
                for diner in self.diners:
                    if diner.getStatus() == index:
                        print("     ",diner)
    #fn takeOrders
    #input: None
    #side-effect: Prompts user for input. Prints user chocie
    #return: None
    def takeOrders(self):
        for diner in self.diners:
            if diner.getStatus() == 1:
                for counter in range(len(Menu.MENU_ITEM_TYPES)):
                    self.menu.printMenuItemsByType(Menu.MENU_ITEM_TYPES[counter])
                    print(" ")
                    selection = int(input(diner.getName()+",Please select a "+Menu.MENU_ITEM_TYPES[counter]+" item."))
                    while selection > self.menu.getNumMenuItemsByType(Menu.MENU_ITEM_TYPES[counter])-1 or selection < 0:
                        selection = int(input(diner.getName() + ",Please select a " + Menu.MENU_ITEM_TYPES[counter] + " item."))
                    order = self.menu.getMenuItem(Menu.MENU_ITEM_TYPES[counter],selection)
                    diner.addtoOrder(order)
                diner.printOrder()
    #ringUpDiners
    #input: None
    #side-effect: Prints meal cost
    #return: None
    def ringUpDiners(self):
        for diner in self.diners:
            if diner.getStatus() == 3:
                print(diner.getName()+", your meal cost $",diner.calculateMealCost())
    #removeDoneDiners
    #input: None
    #return: None
    def removeDoneDiners(self):
        for diner in self.diners:
            if diner.getStatus() == 4:
                print("Thank you "+diner.getName()) #Fix Later
                for num in range(len(self.diners)-1,-1,-1):
                    del self.diners[num]
    #fn advanceDiners
    #input: None
    #return: None
    def advanceDiners(self):
        self.printDinerStatuses()
        self.takeOrders()
        self.ringUpDiners()
        self.removeDoneDiners()
        for diner in self.diners:
            diner.updateStatus()

