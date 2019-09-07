#This class represents the individual items on a menu.

class MenuItem(object):
    def __init__(self,name,foodType,price,description):
        self.name = name
        self.type = foodType
        self.price = price
        self.description = description
    #fn getName
    #input None
    #returns name
    def getName(self):
        return self.name
    #fn setName
    #input: name (str)
    #returns:None
    def setName(self,name):
        if name.isalpha() and name != "":
            self.name = name
        else:
            print("Invalid name.")
    #fn: getType
    #input: none
    #returns: type(str)
    def getType(self):
        return self.type
    #fn setType
    #input: foodType (str)
    #returns:none
    def setType(self,foodType):
        if foodType.isalpha() and foodType != "":
            self.type = foodType
        else:
            print("Invalid type.")
    #fn getPrice
    #input: none
    #returns: price(int)
    def getPrice(self):
        return self.price
    #fn setPrice
    #input: price (int)
    #returns: None
    def setPrice(self,price):
        if price.isnumeric() and price != "":
            self.price = price
        else:
            print("Invalid price.")
    #fn getDescription
    #input: None
    #return: description  (str)
    def getDescription(self):
        return self.description
    #fn setDescription
    #input: description (str)
    #return: None
    def setDescription(self,description):
        if description.isalpha() and description != "":
            self.description = description
        else:
            print("Invalid name.")
    def __str__(self):
        msg = self.name + "("+self.type+")" +":"+"$"+self.price
        msg += "\n" + "   " + self.description
        return msg
    



