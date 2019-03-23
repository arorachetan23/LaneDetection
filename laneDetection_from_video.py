import cv2
import numpy as np

def make_coordinates(image,parameter):
	y1=image.shape[0]
	y2=int(y1*3/5)
	slope=parameter[0]
	intercept=parameter[1]
	x1=int((y1-intercept)/slope)
	x2=int((y2-intercept)/slope)

	return np.array([x1,y1,x2,y2])

def pre_process(img):
	img=np.copy(img)
	imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur=cv2.GaussianBlur(imgray,(5,5),0)
	canny=cv2.Canny(blur,50,150)

	return canny

def detecting_lines(img):
	height=img.shape[0]
	tri=np.array([[(200,height),(1100,height),(550,250)]])
	mask=np.zeros_like(img)
	cv2.fillPoly(mask,tri,255)
	bitwise=cv2.bitwise_and(img,mask)

	lines=cv2.HoughLinesP(bitwise,2,np.pi/180,100,np.array([]), minLineLength=40,maxLineGap=5)
			
	return lines

def average_out(image,lines):
	left_avg=[]
	right_avg=[]
	for line in lines:
		x1,y1,x2,y2=line.reshape(4)
		param=np.polyfit((x1,x2),(y1,y2),1)
		slope=param[0]
		intercept=param[1]
		if slope<0:
			left_avg.append(param)
		else:
			right_avg.append(param)
	left_avg=np.average(left_avg,axis=0)
	right_avg=np.average(right_avg,axis=0)

	left_coordinates=make_coordinates(image,left_avg)
	right_coordinates=make_coordinates(image,right_avg)

	return np.array([left_coordinates,right_coordinates])

def display_lines(image,lines):
	line_image=np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1,y1,x2,y2=line.reshape(4)
			cv2.line(line_image,(x1,y1),(x2,y2),(255,255,0),10)

	added_image=cv2.addWeighted(image,0.8,line_image,1,1)

	return added_image

#capuring video
cap=cv2.VideoCapture("video.mp4")
while(cap.isOpened()):
	_,image=cap.read()

	img=pre_process(image)
	lines=detecting_lines(img)

	avg_lines=average_out(img,lines)

	added_img=display_lines(image,avg_lines) 


	#cv2.imshow("org",image)
#cv2.imshow("gray",imgray)
#cv2.imshow("blur",blur)
#cv2.imshow("canny",canny)
#cv2.imshow("mask",mask)
#cv2.imshow("bitwise",bitwise)
	cv2.imshow("line_image",added_img)
	if cv2.waitKey(1)==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()		
