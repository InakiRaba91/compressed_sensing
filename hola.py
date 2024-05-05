from functools import partial
from manimlib import *

def l1_ball(t, s):
    if 0 <= t <= 1:
        pt = (s * (1 - t), s * t, 0)
    elif 1 < t <= 2:
        pt = (s * (1 - t), s * (2 - t), 0)
    elif 2 < t <= 3:
        pt = (s * (t - 3), s * (2 - t), 0)
    elif 3 < t <= 4:
        pt = (s * (t - 3), s * (t - 4), 0)
    else:
        pt = (0, 0, 0)
    return pt
    
class GraphExample(Scene):
    def construct(self):
        axes = Axes((-7, 7), (-4, 4))
        axes.add_coordinate_labels()

        self.play(Write(axes, lag_ratio=0.01, run_time=1))

        # Axes.get_graph will return the graph of a function
        # l1 = ParametricCurve(partial(l1_ball, s=1), t_range=[0, 4], color=BLUE)        
        # self.add(axes, l1)
        self.wait()
        
        # l1_graph = axes.get_graph(partial(l1_ball, s=1), x_range=[0, 4, 0.01], color=BLUE)
        l1_graph = axes.get_parametric_curve(partial(l1_ball, s=1), x_range=[0, 4], color=BLUE)

        # Axes.get_graph_label takes in either a string or a mobject.
        # If it's a string, it treats it as a LaTeX expression.  By default
        # it places the label next to the graph near the right side, and
        # has it match the color of the graph
        # l1_label = axes.get_graph_label(l1_graph, "|x|_1")

        self.play(
            ShowCreation(l1_graph),
            # FadeIn(l1_label, RIGHT),
        )
        self.wait()

        # # You can use axes.input_to_graph_point, abbreviated
        # # to axes.i2gp, to find a particular point on a graph
        # dot = Dot(color=RED)
        # dot.move_to(axes.i2gp(0, l1_graph))
        # self.play(FadeIn(dot, scale=0.5))

        # # A value tracker lets us animate a parameter, usually
        # # with the intent of having other mobjects update based
        # # on the parameter
        # x_tracker = ValueTracker(2)
        # f_always(
        #     dot.move_to,
        #     lambda: axes.i2gp(x_tracker.get_value(), parabola)
        # )

        # self.play(x_tracker.animate.set_value(4), run_time=3)
        # self.play(x_tracker.animate.set_value(-2), run_time=3)
        # self.wait()


def get_l2_ball(a: float, b: float, ct: np.ndarray, phi: float, color: str):
    return Ellipse(width=2*a, height=2*b, color=color).rotate(phi).shift(ct)

def get_tangent(a: float, b: float, ct: np.ndarray, phi: float, color: str):
    n1 = np.sign(ct[0] * ct[1])
    m = (n1 * math.cos(phi) - math.sin(phi)) / (math.cos(phi) + n1 *  math.sin(phi))
    num = math.sqrt((m * a) ** 2 + b ** 2)
    den = math.cos(phi) - m * math.sin(phi)
    if ct[1] >= 0:
        n2 = abs(den) / den
    else:
        n2 = -abs(den) / den
    p = ct[1] - n1 * ct[0] - n2 * (num / den)
    return DashedLine(start=np.array([- n1 * p, 0, 0]), end=np.array([0, p, 0]), color=color)

def get_centered_segment(m: float, ct: np.ndarray, length: float):
    delta_x = length / math.sqrt(1 + m ** 2)
    delta_y = m * delta_x
    start = ct - np.array([delta_x, delta_y, 0])
    end = ct + np.array([delta_x, delta_y, 0])
    return (start, end)

def get_point_tangency(a: float, b: float, ct: np.ndarray, phi: float):
    n1 = np.sign(ct[0] * ct[1])
    q = (math.cos(phi) + n1 * math.sin(phi)) / (n1 * math.cos(phi) -  math.sin(phi))
    x0 = a ** 2 * q / math.sqrt((a * q) ** 2 + b ** 2)
    y0 = b ** 2 / math.sqrt((a * q) ** 2 + b ** 2)
    x_rot = x0 * math.cos(phi) - y0 * math.sin(phi)
    y_rot = x0 * math.sin(phi) + y0 * math.cos(phi)
    x_shift = x_rot + ct[0]
    y_shift = y_rot + ct[1]
    return np.array([x_shift, y_shift, 0])

def get_tangent(a: float, b: float, ct: np.ndarray, phi: float, color: str):
    n1 = - np.sign(ct[0] * ct[1])
    m = (n1 * math.cos(phi) - math.sin(phi)) / (math.cos(phi) + n1 *  math.sin(phi))
    num = math.sqrt((m * a) ** 2 + b ** 2)
    den = math.cos(phi) - m * math.sin(phi)
    if ct[1] >= 0:
        n2 = abs(den) / den
    else:
        n2 = -abs(den) / den
    p = ct[1] - n1 * ct[0] - n2 * (num / den)
    # line is of the form: y = -n1 * x + p
    pt_tangency = get_point_tangency(a=a, b=b, ct=ct, phi=phi)
    start, end = get_centered_segment(m=-n1, ct=pt_tangency, length=1)
    return DashedLine(start=start, end=end, color=color), Dot(pt_tangency, color=color)
    # return DashedLine(start=np.array([- n1 * p, 0, 0]), end=np.array([0, p, 0]), color=color)

def get_l2_ball_with_tangent(a: float, b: float, ct: np.ndarray, phi: float, color_ellipse: str, color_tangent: str):
    ellipse = get_l2_ball(a=a, b=b, ct=ct, phi=phi, color=color_ellipse)
    tangent, pt_tangency = get_tangent(a=a, b=b, ct=ct, phi=phi, color=color_tangent)
    return VGroup(ellipse, tangent, pt_tangency)

def get_l1_ball(l: float, color: str):
    return Rectangle(width=l, height=l, color=color).rotate(PI/4)

def get_l2_ball_scalled_tangent_to_l1_ball(a: float, b: float, ct: np.ndarray, phi: float, l: float, color: str):
    n1 = - np.sign(ct[0] * ct[1])
    n2 = np.sign(ct[1])
    m = (n1 * math.cos(phi) - math.sin(phi)) / (math.cos(phi) + n1 * math.sin(phi))
    den = math.sqrt((m * a) ** 2 + b ** 2)
    num = (n2 * l - ct[1] + n1 * ct[0]) * (math.cos(phi) - m * math.sin(phi))
    s = num / den      
    return Ellipse(width=2*a*s, height=2*b*s, color=color).rotate(phi).shift(ct)

class CompressedSensing(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Create an ellipse
        a, b = 1, 0.5
        cts = [
             2*RIGHT + 2*UP,
            -2*RIGHT + 2*UP,
            -2*RIGHT - 2*UP,
             2*RIGHT - 2*UP,
        ]

        phi = PI/3
        l = 1
        l2_balls_and_tangents, scaled_ellipses = [], []
        for s in np.linspace(1, 2, 1):
            for ct in cts:
                l2_balls_and_tangents.append(get_l2_ball_with_tangent(a=s * a, b=s * b, ct=ct, phi=phi, color_ellipse=RED, color_tangent=GREEN))
                scaled_ellipses.append(get_l2_ball_scalled_tangent_to_l1_ball(a=a, b=b, ct=ct, phi=phi, l=l, color=YELLOW))
        l2_balls_and_tangents = VGroup(*l2_balls_and_tangents)
        scaled_ellipses = VGroup(*scaled_ellipses)
        l1_ball = get_l1_ball(l=math.sqrt(2*l), color=BLUE)

        # Add the axes and the ellipse to the scene
        self.play(ShowCreation(l2_balls_and_tangents), ShowCreation(l1_ball), ShowCreation(scaled_ellipses))
        self.wait()