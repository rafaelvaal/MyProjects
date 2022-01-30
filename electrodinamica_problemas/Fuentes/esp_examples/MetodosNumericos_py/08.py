from vpython import *
#GlowScript 2.7 VPython
h = 1.0/365.0
m = 1.0/332946.0
ep = 0.017
T = 1.0

G = 4.0*pi*pi
k = G*m
E = -pow(pi*pi*k*k*m/2.0/T/T,1.0/3.0)
L2 = m*k*k*(ep*ep-1)/2.0/E
rmin = L2/(1+ep)/m/k

x1=rmin
y1 = 0.0
vx = 0.0
vy = sqrt(2/m*(E+k/rmin))
display(range=1.2)
Tiempo = label(pos=vec(0,0.65,0), color=color.green,height = 16,text='t = 0.000', box=False)
Sol = sphere(pos=vec(0,0,0), radius=0.1, color=color.yellow)
Tierra = sphere(pos=vec(x1,y1,0), radius=0.05, color=color.blue,make_trail=True)

def fx(x,y):
    r3=pow(x*x+y*y,1.5)
    return(-G/r3*x)

def fy(x,y):
    r3=pow(x*x+y*y,1.5)
    return(-G/r3*y)

x0=x1-vx*h+h*h*fx(x1,y1)/2
y0=y1-vy*h+h*h*fy(x1,y1)/2

for t in range(366):
    rate(50)
    x2=2*x1-x0+h*h*fx(x1,y1)
    y2=2*y1-y0+h*h*fy(x1,y1)
    Tierra.pos=vec(x2,y2,0)
    S = str(t*h)[0:5]
    Tiempo.text='t = ' + S
    sleep(0.1)
    print ('t = ',S, ', x = ',x2,', y = ',y2)
    x0 = x1; x1 = x2
    y0 = y1; y1 = y2