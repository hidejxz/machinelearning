from numpy import *
from tkinter import *
import RegTree as rt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

tree_clf = rt.RegTree()

def redraw(tols, toln):
    redraw.f.clf()
    redraw.a = redraw.f.add_subplot(111)
    if chkBtnVar.get():
        toln = max(toln,2)

        my_tree = tree_clf.create_tree(redraw.rawDat, rt.model_leaf, rt.model_err, 
                                (tols, toln))
        y_hat = tree_clf.create_forecast(my_tree, redraw.testDat, rt.model_tree_eval)
    else:
        my_tree = tree_clf.create_tree(redraw.rawDat, ops = (tols, toln))
        y_hat = tree_clf.create_forecast(my_tree, redraw.testDat)
    #print(type(redraw.rawDat))
    redraw.a.scatter(array(redraw.rawDat)[:,0],array(redraw.rawDat)[:,1] ,s=5)
    redraw.a.plot(redraw.testDat, array(y_hat), linewidth = 2.0)
    redraw.canvas.show()

def getinputs():
    try:
        toln = int(toln_entry.get())
    except:
        toln = 10
        print('enter Interger for tolN')
        toln_entry.delete(0, END)
        toln_entry.insert(0, '10')
    try:
        tols = float(tols_entry.get())
    except:
        tols = 1.0
        print('enter Interger for tolS')
        tols_entry.delete(0, END)
        tols_entry.insert(0, '1.0')
    return toln, tols


def draw_newtree():
    toln, tols = getinputs()
    redraw(tols, toln)

root = Tk()
redraw.f = Figure(figsize = (5,4), dpi = 100)
redraw.canvas = FigureCanvasTkAgg(redraw.f, master = root)
redraw.canvas.show()
redraw.canvas.get_tk_widget().grid(row = 0, columnspan = 3)

Label(root, text='tolN').grid(row = 1, column = 0)
toln_entry = Entry(root)
toln_entry.grid(row = 1, column = 1)
toln_entry.insert(0, '10')

Label(root, text='tolS').grid(row = 2, column = 0)
tols_entry = Entry(root)
tols_entry.grid(row = 2, column = 1)
tols_entry.insert(0, '1.0')

Button(root, text = 'ReDraw', command = draw_newtree).\
    grid(row = 1, column = 2, rowspan = 3)

chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text = 'Model Tree', variable = chkBtnVar)
chkBtn.grid(row = 3, column = 0, columnspan = 2)
redraw.rawDat = mat(rt.load_dataset('sine.txt'))
#print(min(array(redraw.rawDat)[0]))
redraw.testDat = arange(min(redraw.rawDat[:,0]), 
                       max(redraw.rawDat[:,0]), 0.01)
redraw(1.0, 10)
root.mainloop()