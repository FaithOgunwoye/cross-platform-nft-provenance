"""Replot the three benchmark figures from the released CSVs."""

import make_camera_figures
import make_figures


def main():
    make_figures.style()
    make_figures.fig3_heatmap()
    make_camera_figures.style()
    make_camera_figures.fig_scale()
    make_camera_figures.fig_hard()


if __name__ == "__main__":
    main()
