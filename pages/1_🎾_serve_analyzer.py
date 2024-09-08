import streamlit as st
import os
import cv2
import pickle
import mediapipe as mp
import plotly.graph_objects as go
import pandas as pd
import time
import pdb
import numpy as np
from collections import deque
import math

mp_pose = mp.solutions.pose
st.set_page_config(layout="wide")
if 'idx' not in st.session_state:
    st.session_state.idx = 0
if 'idx2' not in st.session_state:
    st.session_state.idx2 = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False


def draw_3d_animation(lands_data, color, frame_idx): #, idx, fw, fh):
    df = lands_data[frame_idx]

    x_mid = (df['x'].min() + df['x'].max()) / 2
    z_mid = (df['z'].min() + df['z'].max()) / 2
    
    x_min, x_max = (lands_data[0]["x"].min(), lands_data[0]["x"].min())
    z_min, z_max = (lands_data[0]["x"].min(), lands_data[0]["x"].min())
    y_min, y_max = (lands_data[0]["x"].min(), lands_data[0]["x"].min())
    for frame in lands_data:
        x_min = frame["x"].min() if frame["x"].min() < x_min else x_min
        x_max = frame["x"].max() if frame["x"].max() > x_max else x_max
        z_min = frame["z"].min() if frame["z"].min() < z_min else z_min
        z_max = frame["z"].max() if frame["z"].max() > z_max else z_max
        y_min = frame["y"].min() if frame["y"].min() < y_min else y_min
        y_max = frame["y"].max() if frame["y"].max() > y_max else y_max

    left_side = [7, 11, 13, 15, 23, 25, 27, 29, 31]
    right_side = [8, 12, 14, 16, 24, 26, 28, 30, 32]
    chest_coordinates = df.loc[[12, 11, 23, 24]]

    # Coletar coordenadas para os dois lados
    df = lands_data[frame_idx]
    frames = []

    # Adicione os frames
    for frame_idx, df in enumerate(lands_data):
        frame_data = []  # Inicialize uma lista vazia para os dados deste frame

        left_coordinates = df.loc[left_side]
        right_coordinates = df.loc[right_side]
        
        # Adicione os pontos
        frame_data.append(
            go.Scatter3d(
                x=left_coordinates['x'], 
                z=left_coordinates['y'], 
                y=left_coordinates['z'], 
                mode='markers', 
                opacity=0.5, 
                marker=dict(color='blue')
            )
        )

        frame_data.append(
            go.Scatter3d(
                x=right_coordinates['x'], 
                z=right_coordinates['y'], 
                y=right_coordinates['z'], 
                mode='markers', 
                marker=dict(color='red')
            )
        )
        
        # Adicione as linhas de conexão
        for connection in mp_pose.POSE_CONNECTIONS:
            from_idx, to_idx = connection
            frame_data.append(
                go.Scatter3d(
                    x=[df.loc[from_idx, 'x'], df.loc[to_idx, 'x']],
                    z=[df.loc[from_idx, 'y'], df.loc[to_idx, 'y']],
                    y=[df.loc[from_idx, 'z'], df.loc[to_idx, 'z']],
                    mode='lines',
                    marker=dict(color='gray')
                )
            )
        
        frame = go.Frame(data=frame_data, name=f'Frame {frame_idx}')
        frames.append(frame)

    # Dados iniciais
    initial_data = frames[0]['data']

    animation_settings = dict(frame=dict(duration=100, redraw=True), fromcurrent=True)
    fig = go.Figure(
        data=initial_data, frames=frames,
        layout=go.Layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play", method="animate", args=[None, animation_settings]),
                        dict(label="Pause", method="animate", args=[
                            [None], dict(frame=dict(duration=0, redraw=False), mode="immediate")
                        ]),
                    ]
                )
            ],
            sliders=[dict(steps= [dict(args=[[f'Frame {i}'], animation_settings],
                                        label=f'Frame {i}',
                                        method="animate") for i, _ in enumerate(frames)],
                        transition=dict(duration=0))]
        )
    )


    fig.update_layout(
        scene=dict(
                xaxis=dict(range=[x_min, x_max], showticklabels=True, showgrid=True, zeroline=False, showline=False),
                zaxis=dict(range=[y_min, y_max], showticklabels=True, showgrid=True, zeroline=False, showline=False),
                yaxis=dict(range=[z_min, z_max], showticklabels=True, showgrid=True, zeroline=False, showline=False),
                camera=dict(eye=dict(x=x_mid, y=-2, z=0.1)),
                aspectmode='cube',
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=700,
            showlegend=False,
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Frame:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            # "x": 0.1,
            # "y": 0
        }]
    )
                
    return fig
    

def draw_3d_representation(fig, lands_data, color, frame_idx): #, idx, fw, fh):
    df = lands_data[frame_idx]
    shadow_lands = lands_data[:frame_idx][-10:]
    
    x_min, x_max = (lands_data[0]["x"].min(), lands_data[0]["x"].max())
    y_min, y_max = (lands_data[0]["y"].min(), lands_data[0]["y"].max())
    z_min, z_max = (lands_data[0]["z"].min(), lands_data[0]["z"].max())
    for frame in lands_data:
        x_min = frame["x"].min() if frame["x"].min() < x_min else x_min
        x_max = frame["x"].max() if frame["x"].max() > x_max else x_max
        z_min = frame["z"].min() if frame["z"].min() < z_min else z_min
        z_max = frame["z"].max() if frame["z"].max() > z_max else z_max
        y_min = frame["y"].min() if frame["y"].min() < y_min else y_min
        y_max = frame["y"].max() if frame["y"].max() > y_max else y_max
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    left_side = [7, 11, 13, 15, 23, 25, 27, 29, 31]
    right_side = [8, 12, 14, 16, 24, 26, 28, 30, 32]
    chest_coordinates = df.loc[[12, 11, 23, 24]]

    # Coletar coordenadas para os dois lados
    left_coordinates = df.loc[left_side]
    right_coordinates = df.loc[right_side]

    # Adicionar pontos com cores diferentes para cada lado
    fig.add_trace(go.Scatter3d(x=left_coordinates['x'], z=left_coordinates['y'], y=left_coordinates['z'], mode='markers', opacity=0.5, marker=dict(color=color)))
    fig.add_trace(go.Scatter3d(x=right_coordinates['x'], z=right_coordinates['y'], y=right_coordinates['z'], mode='markers', marker=dict(color=color)))


    # if len(shadow_lands) > 1:
    #     bp = st.session_state["body_part"]
    #     x_coords = [frame.loc[bp, "x"] for frame in shadow_lands]
    #     y_coords = [frame.loc[bp, "y"] for frame in shadow_lands]
    #     z_coords = [frame.loc[bp, "z"] for frame in shadow_lands]
    #     n = len(shadow_lands)
    #     opacities = np.linspace(0.2, 1.0, n) 
    #     for i in range(n):
    #         fig.add_trace(go.Scatter3d(
    #             x=[x_coords[i]], 
    #             z=[y_coords[i]], 
    #             y=[z_coords[i]], 
    #             mode='markers', 
    #             marker=dict(
    #                 color='green',
    #                 opacity=opacities[i]  # Define a opacidade do marcador
    #             )
    #         ))

    # Adiciona a linha do eixo vertical ao gráfico


    fig.add_trace(go.Mesh3d(
        x=chest_coordinates['x'],
        z=chest_coordinates['y'],
        y=chest_coordinates['z'],
        i=[0, 0, 1],
        j=[1, 2, 3],
        k=[2, 3, 3],
        color=color, 
        opacity=0.3
    ))

    # Adicionar linhas (conexões entre os pontos) 
    for connection in mp_pose.POSE_CONNECTIONS:
        from_idx, to_idx = connection
        fig.add_trace(go.Scatter3d(
            x=[df.loc[from_idx, 'x'], df.loc[to_idx, 'x']],
            z=[df.loc[from_idx, 'y'], df.loc[to_idx, 'y']],
            y=[df.loc[from_idx, 'z'], df.loc[to_idx, 'z']],
            mode='lines',
            marker=dict(color='gray')
        ))
    
    # pdb.set_trace()
    fig.update_layout(                                                                              
                scene=dict(
                xaxis=dict(range=[x_min, x_max], showticklabels=True, showgrid=True, zeroline=False, showline=False),
                yaxis=dict(range=[z_min, z_max], showticklabels=True, showgrid=True, zeroline=False, showline=False),
                zaxis=dict(range=[y_min, y_max], showticklabels=True, showgrid=True, zeroline=False, showline=False),
                camera=dict(eye=dict(x=0, y=-2, z=0.5)),
                aspectmode='manual',
                aspectratio=dict(x=1, y=0.5, z=1.5),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600,
            showlegend=False)
            
    return fig


def draw_sidebar(landmarks, landmarks2):
    global z_scale, z_scale_ref
    expander_frame = st.sidebar.expander("🎛️ Frame Controller", True)
    slider_play = expander_frame.empty()
    slider_ref_play = expander_frame.empty()
    col3, col4 = expander_frame.columns([0.5, 1])
    col7, col8 = st.sidebar.columns([0.5, 1])

    if col3.button('-1'):
        st.session_state.idx = st.session_state.idx - 1 if st.session_state.idx > 0 else 0
    if col4.button('+1'):
        st.session_state.idx = st.session_state.idx + 1 if st.session_state.idx < len(landmarks) else len(landmarks) - 1
    if col3.button('-1 R'):
        st.session_state.idx2 = st.session_state.idx2 - 1 if st.session_state.idx2 > 0 else 0
    if col4.button('+1 R'):
        st.session_state.idx2 = st.session_state.idx2 + 1 if st.session_state.idx2 < len(landmarks2) else len(landmarks2) - 1
    
    if col3.button('Voltar'):
        st.session_state.is_playing = False
        st.session_state.idx = st.session_state.idx - 1 if st.session_state.idx > 0 else 0
        st.session_state.idx2 = st.session_state.idx2 - 1 if st.session_state.idx2 > 0 else 0
    if col4.button('Avançar'):
        st.session_state.is_playing = False
        st.session_state.idx = st.session_state.idx + 1 if st.session_state.idx < len(landmarks) else len(landmarks) - 1
        st.session_state.idx2 = st.session_state.idx2 + 1 if st.session_state.idx2 < len(landmarks2) else len(landmarks2) - 1
    
    if col7.button('▶️ Play'):
        st.session_state.is_playing = True
    if col8.button('⏸️ Pause'):
        st.session_state.is_playing = False


    st.session_state["idx"] = slider_play.slider('Main Video', 0, len(landmarks) - 1, st.session_state["idx"])
    st.session_state["idx2"] = slider_ref_play.slider('Ref Video', 0, len(landmarks2) - 1, st.session_state.idx2)
    
    # z_scale = expander_scalers.slider('Z Scaler', 0.5, 1.5, step=0.1)
    # z_scale_ref = expander_scalers.slider('Z Scaler Ref', 0.5, 1.5, step=0.1)


def draw_shadow(frame, lands_data, idx, bp, h, w):
    idx0 = max(idx - 10, 0)
    line_points = [(int(i.landmark[bp].x * w), int(i.landmark[bp].y * h)) for i in lands_data[idx0:idx]]
    for i in range(1, len(line_points)):
        thickness = int(np.sqrt(10 / float(i + 1)) * 2.5)
        cv2.line(frame, line_points[i - 1], line_points[i], (50, 255, 255), thickness)
    return frame


def find_angle(frame, landmarks, idx, p1, p2, p3):
    land = landmarks[idx].landmark
    h, w, c = frame.shape
    # landmarks[idx].landmark
    x1, y1 = (land[p1].x, land[p1].y)
    x2, y2 = (land[p2].x, land[p2].y)
    x3, y3 = (land[p3].x, land[p3].y)

    angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
    position = (int(x2 * w + 10), int(y2 * h +10))
    frame = cv2.putText(frame, str(int(angle)), position, 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
    return frame


def find_angle2(frame, landmarks, idx, p1, p2, p3):
    land = landmarks[idx].landmark
    h, w, c = frame.shape
    x1, y1 = (land[p1].x, land[p1].y)
    x2, y2 = (land[p2].x, land[p2].y)
    x3, y3 = (land[p3].x, land[p3].y)

    a = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    b = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)
    c = math.sqrt((x3 - x1)**2 + (y3 - y1)**2)

    cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)
    angle = math.degrees(math.acos(cos_gamma))
    position = (int(x2 * w + 10), int(y2 * h +10))
    frame = cv2.putText(frame, str(int(angle)), position, 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
    return frame, angle


def calc_pos_diff(pro_frame, my_lands):
    frame_diff = []
    for i in range(len(pro_frame)):
        pro_point = [pro_frame.loc[i, 'x'], pro_frame.loc[i, 'y'], pro_frame.loc[i, 'z']]
        your_point = [my_lands.loc[i, 'x'], my_lands.loc[i, 'y'], my_lands.loc[i, 'z']]
        dist = np.linalg.norm(np.array(pro_point) - np.array(your_point))
        frame_diff.append(dist) 
    right_arm = [12, 14, 16]
    left_arm = [11, 13, 15]
    right_leg = [24, 26, 28]
    left_leg = [23, 25, 27]
    
    right_arm_diff = np.array([frame_diff[i] for i in right_arm]).sum()
    left_arm_diff = np.array([frame_diff[i] for i in left_arm]).sum()
    right_leg_diff = np.array([frame_diff[i] for i in right_leg]).sum()
    left_leg_diff = np.array([frame_diff[i] for i in left_leg]).sum()
    # print(right_arm_diff, left_arm_diff, right_leg_diff, left_leg_diff)
    return frame_diff


@st.cache_data()
def load_data(video, video2, render_3d):
    prefix = "norm_" if render_3d else ""
    landmarks_file = f"landmarks/{prefix}landmarks_{video.split('.')[0]}.pickle"
    with open(landmarks_file, 'rb') as f:
        lands_data = pickle.load(f)

    landmarks_file = f"landmarks/{prefix}landmarks_{video2.split('.')[0]}.pickle"
    with open(landmarks_file, 'rb') as f:
        lands_data2 = pickle.load(f)
    return lands_data, lands_data2


@st.cache_resource
def load_videos(video, video2):
    cap = {
            "my": cv2.VideoCapture(f'serve/{video}'),
            "pro": cv2.VideoCapture(f'serve/{video2}')
        }
    return cap

video_files = [i for i in os.listdir('serve') if i[0] != '.']
video_files.sort()
video = st.sidebar.selectbox("Selecione um vídeo:", video_files)
video2 = st.sidebar.selectbox("Selecione um vídeo de ref:", video_files, index=len(video_files)-1)

if video and video2:
    render_3d = st.checkbox("Render 3D?")
    body_parts = {"right upper": 16, 
                  "left upper": 15,
                  "right lower": 28,
                  "left lower": 27}
    angle_parts = {"right arm": (11, 13, 15),
                   "left arm": (12, 14, 16),
                   "right leg": (24, 26, 28),
                   "left leg": (23, 25, 27),
                   "right_torso": (12, 24, 26),
                   "left_torso": (11, 23, 25)}
    # bp = st.selectbox("Shadow on", body_parts.keys(), 0)
    ap = st.multiselect("Angle on", angle_parts.keys(), ["right arm"])

    lands_data, lands_data2 = load_data(video, video2, render_3d)
    cap = load_videos(video, video2)
    draw_sidebar(lands_data, lands_data2) 
    
    col1, col2, col3 = st.columns([0.4, 0.4, 0.5])
    ph = col1.empty()
    cont = ph.container() 
    ph2 = col2.empty()
    cont2 = ph2.container()
    ph3 = col3.empty()
    cont3 = ph3.container()

    # st.session_state["body_part"] = body_parts[bp]
    h, w = cap["my"].get(cv2.CAP_PROP_FRAME_HEIGHT), cap["my"].get(cv2.CAP_PROP_FRAME_WIDTH)
    h2, w2 = cap["pro"].get(cv2.CAP_PROP_FRAME_HEIGHT), cap["pro"].get(cv2.CAP_PROP_FRAME_WIDTH)

    cap["my"].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx"])
    cap["pro"].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx2"])
    ret, frame = cap["my"].read()        
    ret2, frame2 = cap["pro"].read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    if not st.session_state["is_playing"]:
        if not render_3d:
            for i in ap:
                ap_tup = angle_parts[i]
                frame = draw_shadow(frame, lands_data, st.session_state["idx"], ap_tup[2], h, w)            
                frame2 = draw_shadow(frame2, lands_data2, st.session_state["idx2"], ap_tup[2], h2, w2)
                
                frame, a1 = find_angle2(frame, lands_data, st.session_state["idx"], ap_tup[0], ap_tup[1], ap_tup[2])
                frame2, a2 = find_angle2(frame2, lands_data2, st.session_state["idx2"], ap_tup[0], ap_tup[1], ap_tup[2])
                col3.metric(f"{i}", round((a2 - a1)*100)/100)
            cont.image(frame, channels="RGB")
            cont2.image(frame2, channels="RGB")
        else:
            fig = draw_3d_representation(go.Figure(), lands_data, "blue", st.session_state["idx"])
            fig2 = draw_3d_representation(go.Figure(), lands_data2, "red", st.session_state["idx2"])
            cont.plotly_chart(fig, use_container_width=True) 
            cont2.plotly_chart(fig2, use_container_width=True) 

    while st.session_state["is_playing"]:
        st.session_state["idx"] += 1 if st.session_state["idx"] < len(lands_data) - 1 else 0
        st.session_state["idx2"] += 1 if st.session_state["idx2"] < len(lands_data2) - 1 else 0
       
        cap["my"].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx"])
        cap["pro"].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx2"])
        ret, frame = cap["my"].read()
        ret2, frame2 = cap["pro"].read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        if not render_3d:
            with ph3.container() as p:
                for i in ap:
                    ap_tup = angle_parts[i]
                    frame = draw_shadow(frame, lands_data, st.session_state["idx"], ap_tup[2], h, w)            
                    frame2 = draw_shadow(frame2, lands_data2, st.session_state["idx2"], ap_tup[2], h2, w2)
                    frame, a1 = find_angle2(frame, lands_data, st.session_state["idx"], ap_tup[0], ap_tup[1], ap_tup[2])
                    frame2, a2 = find_angle2(frame2, lands_data2, st.session_state["idx2"], ap_tup[0], ap_tup[1], ap_tup[2])
                    st.metric(f"{i}", round((a2 - a1)*100)/100)

            with ph.container() as p:
                st.image(frame, channels="RGB")
            with ph2.container() as p:
                st.image(frame2, channels="RGB")
        
        else:
            fig = draw_3d_representation(go.Figure(), lands_data, "blue", st.session_state["idx"])
            fig2 = draw_3d_representation(go.Figure(), lands_data2, "red", st.session_state["idx2"])

            ph.plotly_chart(fig, use_container_width=True)  
            ph2.plotly_chart(fig2, use_container_width=True)  


    
        
        
