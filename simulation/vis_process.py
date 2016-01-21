import fcntl
import visualiser
import sys
import os

def readpolydata(vtkfile='/dev/shm/visvtk.vtk',
                 lockfile='/dev/shm/visvtk.lock'):
    lockfd = open(lockfile)
    try:
        fcntl.lockf(lockfd, fcntl.LOCK_SH)
        polyd = visualiser.readpolydata(vtkfile)
    finally:
        fcntl.lockf(lockfd, fcntl.LOCK_UN)
    return polyd

def main(lockfkey="visvtk"):
    def timercallback(vtkRenWinInt, event):
        visualiser.clear_window()
        lockf = "/dev/shm/%s.lock" % lockfkey
        rootf, ext = os.path.splitext(lockf)
        vtkf = rootf + ".vtk"
        polyd = readpolydata(vtkf, lockf)
        visualiser.show_axes(scale=1)
        visualiser.visualise(polyd)
        vtkRenWinInt.GetRenderWindow().Render()

    visualiser.show_window(timer_callback=timercallback)


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()
