from MenuItem import MenuItem

#This class represents the Diner object.
class Diner(object):
    STATUSES = ["Seated","Ordering","Eating","Paying","Leaving"]
    def __init__(self,name):
        self.name = name
        self.order = []
        self.status = 0
    #fn getName
    #input: none
    #return: name (str)
    def getName(self):
        return self.name
    #fn setName
    #input: name (str)
    #return: None
    def setName(self,name):
        if name.isalpha() and name != "":
            self.name = name
        else:
            print("Invalid name.")
    #fn getOrder
    #input: none
    #return: order (List of menuitem objects)
    def getOrder(self):
        return self.order
    #fn setOrder
    #input: order (MenuItem Object)
    #return: none
    def setOrder(self,order):
        for item in order:
            if item is MenuItem:
                self.order.append(item)
    #fn getStatus
    #input: none
    #return: status (int)
    def getStatus(self):
        return self.status
    #fn setStatus
    #input: status (int)
    #return: None
    def setStatus(self,status):
        if status.isnumeric() and status != "":
            self.status = status
    #fn updateStatus
    #input: None
    #return: None
    def updateStatus(self):
        self.status += 1
    #fn addtoOrder
    #input: MenuItem Object
    #return: None
    def addtoOrder(self,MenuItem):
        self.order.append(MenuItem)
    #fn printOrder
    #input: None
    #side-effect: Prints item
    #return: None
    def printOrder(self):
        print(self.getName()+" ordered:")
        for item in self.order:
            print("-",item)
    #fn calculateMealCost
    #input: None
    #return: total (int)
    def calculateMealCost(self):
        total = 0
        for item in self.order:
            price = float(item.getPrice())
            total += price
        return total
    def __str__(self):
        msg = "Diner "+self.name+" is currently "+self.STATUSES[self.status]
        return msg


