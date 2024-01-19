from PIL import Image
import cv2
import streamlit as st
import argparse
from demo import main
from TDDFA import TDDFA
from FaceBoxes import FaceBoxes
import yaml
from utils.tddfa_util import str2bool
from utils.functions import draw_landmarks
st.title("Generate user-specific face model")

st.write("")
#提示上传图片
file_up = st.file_uploader("请上传一张人脸图片(.jpg)", type = "jpg")


parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
parser.add_argument('--onnx', action='store_true', default=False)
opt = parser.parse_args()

dense_flag = True
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
face_boxes = FaceBoxes()
tddfa = TDDFA(gpu_mode=False, **cfg)


#上传文件后开始检测
if file_up is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):


                st.sidebar.image(file_up)
                picture = Image.open(file_up)
                print(file_up.name)
                picture = picture.save(f'./data/show/1.jpg')
                img = cv2.imread('data/show/1.jpg')
                img = img[..., ::-1]  # RGB -> BGR
                boxes = face_boxes(img)
                param_lst, roi_box_lst = tddfa(img, boxes)
                ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
                draw_landmarks(img, ver_lst, dense_flag=dense_flag)
                img2 = cv2.imread('2.png')
                img2 = cv2.resize(img2,(0,0),fx=0.5,fy=0.5)
                cv2.imwrite("3.png",img2)
                st.image('3.png',width=300)

                opt.img_fp = f'./data/show/1.jpg'
                opt.opt = "obj"
                #print(opt)
else:
     is_valid = False

if is_valid:
    main(opt)