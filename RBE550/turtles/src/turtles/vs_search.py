import turtle as tur
import numpy as np

# static datum version
def draw_victor_sierra(radius, heading, t = tur.Turtle()):
    t.setheading(heading + 90)
    for i in range(9):
        t.forward(radius)
        t.right(120 if (i+1) % 3 != 0 else 0)
        t.dot(5)

# moving datum version (for funsies)
def draw_victor_sierra_drift(radius, heading, drift_func, datum_tur, drift_tur):
    base_tur = datum_tur.clone()
    base_tur.speed(0)
    base_tur.hideturtle()
    base_tur.pencolor("gray")
    draw_victor_sierra(radius, heading, base_tur)

    datum_tur.pencolor("green")
    datum_tur.color("green")

    drift_tur.setheading(heading + 90)

    helper_tur = drift_tur.clone()
    helper_tur.hideturtle()
    helper_tur.penup()
    #helper_tur.pencolor("cyan")
    for i in range(9):
        (dx, dy) = drift_func(i)
        datum_tur.setheading(np.rad2deg(np.atan2(dy, dx)))
        datum_tur.forward(np.linalg.norm((dx, dy)))
        datum_tur.dot(5)

        match(i%3):
            case 0: # first leg: go forward + drift, turn 120deg
                helper_tur.forward(radius)
                (x, y) = helper_tur.pos()
                helper_tur.goto(x + dx, y + dy)
                helper_tur.right(120)
                drift_tur.pencolor("black")
            case 1: # second leg: go forward + drift
                helper_tur.forward(radius)
                (x, y) = helper_tur.pos()
                helper_tur.goto(x + dx, y + dy)
                drift_tur.pencolor("black")
            case 2: # third leg: go to datum
                helper_tur.setheading(helper_tur.towards(datum_tur.pos()))
                helper_tur.forward(helper_tur.distance(datum_tur.pos()))
                drift_tur.pencolor("red")

        drift_tur.setheading(drift_tur.towards(helper_tur.pos()))
        drift_tur.forward(drift_tur.distance(helper_tur.pos()))
        drift_tur.dot(5)

draw_victor_sierra(200, 0)

# more interesting patterns
#datum_tur = tur.Turtle()
#drift_tur = tur.Turtle()
#draw_victor_sierra_drift(200, 0, lambda t: (20, 40*np.cos(t*2*np.pi/9)), datum_tur, drift_tur)
#draw_victor_sierra_drift(200, 0, lambda t: (t*t, 0), datum_tur, drift_tur)
input("Done")