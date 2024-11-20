import os
from skimage import io
from Integration import *
import streamlit as st

directory_path = "./Fog detection testcases"
images = os.listdir(directory_path)
image_name = st.sidebar.selectbox("Select an image:",["1.jpeg", "2.jpeg", "4.jpeg", "f1.jpg", "f3.jpg", "f5.jpg","f6.jpg","f7.jpg","f8.jpg","f9.jpg", "s1.jpeg", "s2.jpeg","s3.jpeg"
,"s4.jpeg","s5.jpg","s6.jpg","s7.jpg","s8.jpg","f10.jpg", "f11.jpg","f15.jpg", "f2.jpg"], index=0)


@st.cache
def pipeline_execution():
    t1=rgb2gray(img)
    res=SunnyImageDetection(t1)
    if not res:
        yVanishing = VanishingPointDetection(t1)
        out1=IterativeThresholding(t1*255)
        yAvg =SkyRoadLimitHeight(t1,out1)
        fogginess = abs(yVanishing - yAvg)/ yVanishing
        degree_of_fog = round(fogginess*100,3)
        se = 15
        darkChannel = DarkChannelPrior(img, se)
        atmosphericLight = AtmosphericLightEstimation(darkChannel, img)
        transmission_map = TransmissionEstimation(img, atmosphericLight, se)
        refined_img_guided = SoftMatting(img, transmission_map,'Guided')
        refined_img_gaussian = SoftMatting(img, transmission_map,'Gaussian')
        refined_img_built_in_bilateral = SoftMatting(img, transmission_map,'Built in Bilateral')
        defogged_img_guided = RecoverSceneRadiance(img, atmosphericLight, refined_img_guided, t0 = 0.1)
        defogged_img_gaussian = RecoverSceneRadiance(img, atmosphericLight, refined_img_gaussian, t0 = 0.25)
        defogged_img_built_in_bilateral = RecoverSceneRadiance(img, atmosphericLight, refined_img_built_in_bilateral, t0 = 0.25)
        white_patch_img  = white_patch(defogged_img_guided)
        gray_world_img = gray_world(defogged_img_guided)
        ground_truth_img_mean = ground_truth(defogged_img_guided, 50, 50, 'mean')
        ground_truth_img_max = ground_truth(defogged_img_guided, 50, 50, 'max')
        
        return res, yVanishing,yAvg, degree_of_fog, defogged_img_guided, defogged_img_gaussian, defogged_img_built_in_bilateral,white_patch_img,gray_world_img, ground_truth_img_mean, ground_truth_img_max
    else:
        return res , 0,0,0,np.zeros([6,6]),np.zeros([6,6]),np.zeros([6,6]),np.zeros([6,6]),np.zeros([6,6]),np.zeros([6,6]),np.zeros([6,6])

if image_name:
    img = io.imread(directory_path+"/"+image_name)
    st.header("Fog Detection and Removal")
    st.subheader("Selected Image:")
    st.image(img, channels="RGB")
    st.markdown("---")
    res,yVanishing,yAvg, degree_of_fog, defogged_img_guided, defogged_img_gaussian, defogged_img_bilateral,white_patch_img,gray_world_img, ground_truth_img_mean, ground_truth_img_max= pipeline_execution()
    with st.spinner('Detecting Fog...'):
        if res:
            st.subheader("No Fog Detected")
            st.write("The image is sunny")
        else:
            st.subheader("Fog Detected")
            st.write("The image is foggy")
            st.write("The degree of fog is: ", degree_of_fog,"%")
            st.write("The height of the vanishing point is: ", yVanishing)
            st.write("The height of the average sky road limit is: ", yAvg)
            st.markdown("---")
            with st.spinner('Removing Fog...'):
                st.subheader("Defogged Image using Guided Filter:")
                st.image(defogged_img_guided,clamp=True, channels='RGB')
                st.markdown("---")
                st.subheader("Defogged Image using Gaussian Filter:")
                st.image(defogged_img_gaussian,clamp=True, channels='RGB')
                st.markdown("---")
                st.subheader("Defogged Image using Bilateral Filter:")
                st.image(defogged_img_bilateral,clamp=True, channels='RGB')
                st.markdown("---")
                st.subheader("Defogged Image using White Patch:")
                st.image(white_patch_img,clamp=True, channels='RGB')
                st.markdown("---")
                st.subheader("Defogged Image using Gray World:")
                st.image(gray_world_img,clamp=True, channels='RGB')
                st.markdown("---")
                st.subheader("Defogged Image using Ground Truth (mean):")
                st.image(ground_truth_img_mean,clamp=True, channels='RGB')
                st.markdown("---")
                st.subheader("Defogged Image using Ground Truth (max):")
                st.image(ground_truth_img_max,clamp=True, channels='RGB')
                st.markdown("---")
                
        