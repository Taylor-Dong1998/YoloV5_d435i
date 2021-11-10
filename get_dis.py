import pyrealsense2 as rs
import numpy as np
import cv2

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

def get_coordinates():
    cv2.namedWindow("depth_frame")
    cv2.setMouseCallback("depth_frame", show_distance)
    pc = rs.pointcloud()
    points = rs.points()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    pipe_profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    n = 0
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        img_color = np.asanyarray(color_frame.get_data())
        img_depth = np.asanyarray(depth_frame.get_data())

        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        tex = np.asanyarray(points.get_texture_coordinates())

        i = 640 * point[1] + point[0]  # as point（200，200）
        # print('vtx depth: ', [np.float(vtx[i][0]), np.float(vtx[i][1]), np.float(vtx[i][2])])
        cv2.circle(img_color, point, 2, [0, 255, 0], thickness=-1)
        cv2.putText(img_color, "Dis:" + str(img_depth[point[1], point[0]]), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    [255, 0, 255])
        cv2.putText(img_color, "X:" + str(np.float64(vtx[i][0])*1000), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 255])
        cv2.putText(img_color, "Y:" + str(np.float64(vtx[i][1])*1000), (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 255])
        cv2.putText(img_color, "Z:" + str(np.float64(vtx[i][2])*1000), (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 255])
        cv2.imshow('depth_frame', img_color)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    point = (400, 300)
    get_coordinates()



