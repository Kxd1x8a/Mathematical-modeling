from PyQt5.QtWidgets import QApplication, QMainWindow
from ui_mainwindow import Ui_MainWindow
import sys
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtGui import QPixmap
from math import sin, cos, pi
from scipy import integrate
from PIL import Image
import os

GRAVITY = 9.80665


class TrajectorySolver:
    def __init__(self, wind_speed, mass, spring_constant, launch_angle, initial_velocity, time_limit=20):
        self.spring_constant = spring_constant
        self.launch_angle = launch_angle
        self.mass = mass
        self.wind_speed = wind_speed
        self.initial_velocity = initial_velocity
        self.initial_velocity_x = self.initial_velocity * cos(self.launch_angle)
        self.initial_velocity_y = self.initial_velocity * sin(self.launch_angle)
        self.spring_to_mass_ratio = self.spring_constant / self.mass
        self.time_points = np.arange(0, time_limit, 0.0005)

    def x_model(self, state, time, wind_speed):
        position_x, velocity_x = state
        dxdt = velocity_x
        dvxdt = -self.spring_to_mass_ratio * velocity_x + wind_speed
        return [dxdt, dvxdt]

    def y_model(self, state, time):
        position_y, velocity_y = state
        dydt = velocity_y
        dvydt = -GRAVITY - self.spring_to_mass_ratio * velocity_y
        return [dydt, dvydt]

    def solve_x(self):
        x_trajectory = integrate.odeint(self.x_model, [0, self.initial_velocity_x], self.time_points,
                                        args=(self.wind_speed,))
        return x_trajectory

    def solve_y(self):
        y_trajectory = integrate.odeint(self.y_model, [0, self.initial_velocity_y], self.time_points)
        return y_trajectory


class ProjectileMotionApp(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.init_buttons()

    def init_buttons(self):
        sliders_line_edits = [
            (self.horizontalSlider_9, self.lineEdit_5),
            (self.horizontalSlider_13, self.lineEdit_4),
            (self.horizontalSlider_14, self.lineEdit_3),
            (self.horizontalSlider_15, self.lineEdit_2),
            (self.horizontalSlider_16, self.lineEdit),
            (self.horizontalSlider_10, self.lineEdit_6),
            (self.horizontalSlider_17, self.lineEdit_7),
        ]

        for slider, line_edit in sliders_line_edits:
            slider.valueChanged[int].connect(lambda value, le=line_edit: self.update_value(value, le))

        self.pushButton.clicked.connect(self.launch_projectile)
        self.pushButton_2.clicked.connect(self.plot_trajectory)

    def update_value(self, value, line_edit):
        line_edit.setText(str(value))

    def launch_projectile(self):
        try:
            wind_speed = float(self.lineEdit_3.text())
            mass = float(self.lineEdit.text())
            spring_constant = float(self.lineEdit_2.text())
            launch_angle = float(self.lineEdit_4.text()) * (pi / 180)
            launch_angle_degrees = float(self.lineEdit_4.text())
            initial_velocity = float(self.lineEdit_5.text())

            trajectory_solver = TrajectorySolver(wind_speed, mass, spring_constant, launch_angle, initial_velocity)
            x_trajectory = trajectory_solver.solve_x()
            y_trajectory = trajectory_solver.solve_y()
            x_positions, y_positions = self.extract_valid_positions(x_trajectory, y_trajectory)

            self.plot_trajectory(x_positions, y_positions, initial_velocity, launch_angle_degrees)

            flying_time = trajectory_solver.time_points[len(y_positions) - 1]
            self.lineEdit_8.setText(str(round(flying_time, 2)))

        except ZeroDivisionError as e:
            print('Error: The mass of the body cannot be zero!', e)

    def extract_valid_positions(self, x_trajectory, y_trajectory):
        x_positions, y_positions = [], []

        for j in range(1, len(y_trajectory)):
            if y_trajectory[j][0] > 0:
                x_positions.append(x_trajectory[j][0])
                y_positions.append(y_trajectory[j][0])
            else:
                break
        return x_positions, y_positions

    def plot_trajectory(self, x_positions, y_positions, initial_velocity, launch_angle_degrees):
        def resize_image(image_path, fixed_width):
            img = Image.open(image_path)
            width_percent = (fixed_width / float(img.size[0]))
            height_size = int((float(img.size[0]) * float(width_percent)) / 2)
            new_image = img.resize((fixed_width, height_size))
            new_image.save('graph.png')

        plt.figure(figsize=(20, 10))
        plt.plot(x_positions, y_positions)
        plt.xlim([-20, np.max(x_positions) * 1.1])
        plt.ylim([-1, np.max(y_positions) * 1.1])
        plt.xlabel('X', fontsize=24)
        plt.ylabel('Y', fontsize=24)
        plt.title(f'Initial Velocity = {int(initial_velocity)}, Launch Angle = {launch_angle_degrees}', fontsize=32)
        plt.grid(True)
        plt.savefig('graph.png')

        if self.radioButton.isChecked():
            plt.savefig('graph.png')
            resize_image('graph.png', 1000)
            pixmap = QPixmap('graph.png')
            self.label_3.setPixmap(pixmap)
        else:
            plt.savefig('graph.png')
            resize_image('graph.png', 1000)
            pixmap = QPixmap('graph.png')
            self.label_3.setPixmap(pixmap)
            os.remove('graph.png')

    def plot_trace(self):
        try:
            self.plot_trace(float(self.lineEdit_6.text()), float(self.lineEdit_7.text()))
        except ValueError as e:
            print('Error:', e)


def main():
    app = QApplication(sys.argv)
    projectile_app = ProjectileMotionApp()
    projectile_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
