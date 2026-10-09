# 2. Computer vision foundations

Previous: [intuition](01_world_models_eli5.md). Next: [representation learning](03_representation_learning.md). This chapter rebuilds the connection from a camera measurement to a useful state estimate.

## ELI10: an image is an array

A digital image stores samples of light intensity. An RGB image has red, green, and blue channels; grayscale has one intensity channel. A NumPy image often has shape ((H,W,C)). PyTorch commonly uses ((B,C,H,W)); video adds time (T), with conventions differing by model.

For example, ((480,640,3)) contains 921,600 channel values. Integers 0–255 are common for storage; neural networks usually use floating-point normalization. OpenCV loads color in **BGR** order.

```python
import cv2, numpy as np, torch
bgr = np.zeros((64, 96, 3), dtype=np.uint8)
bgr[20:30, 40:50] = (0, 0, 255)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
assert x.shape == (1, 3, 64, 96)
```

The same integer can mean color intensity, a segmentation class, or metric depth; inspect semantics, not just shape. A grayscale conversion is a weighted channel combination, not recovery of 3D shape.

**Check:** What happens if a red glass is accidentally passed to an RGB encoder as BGR?

## Classical image operations: problem → mechanism → model connection

The table is a compact first pass. Each operation solves a different problem; its output alone does not establish physical understanding. The snippets use the image above unless stated otherwise.

| Concept and everyday analogy | Internal principle and concrete code | World-model connection and misconception | Check |
|---|---|---|---|
| Transformations: moving a photograph on a desk | A coordinate map samples source pixels; $\tilde p'=A\tilde p$ for affine transforms. `cv2.resize(bgr,(48,32))`; `cv2.warpAffine(bgr,np.float32([[1,0,5],[0,1,0]]),(96,64))` | Resize/crop changes apparent position; transform camera intrinsics consistently. Cropping is not an object action. | What changes in calibration after resize? |
| Filtering: average neighboring measurements to suppress noise | $(I*K)_{ij}=\sum_{u,v}K_{uv}I_{i-u,j-v}$. `cv2.GaussianBlur(gray,(5,5),1)` | Reduces noise but can erase small moving objects. A convolution is not necessarily a learned feature. | Why blur before noisy edge extraction? |
| Edges: outline a door's brightness boundary | Gradients estimate intensity change: $\nabla I=(I_x,I_y)$. `cv2.Canny(gray,50,150)` | Boundaries can support geometry; texture also makes edges. An edge is not guaranteed to be an object boundary. | Is a shadow edge a wall? |
| Contours: trace a silhouette | Connected boundary curves of a binary region. `mask=(gray>30).astype(np.uint8)*255`; `cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)` | Useful for toy tracking; occlusion and threshold changes alter identity. | Can two touching objects share a contour? |
| Features: distinctive corners as landmarks | Corners have gradients in two directions; descriptors permit matching. `cv2.goodFeaturesToTrack(gray,20,.01,3)`; `cv2.ORB_create().detectAndCompute(gray,None)` | Helps motion and SLAM; repetitive patterns make ambiguous matches. | Why is a blank wall hard to track? |
| Tracking: follow one car over frames | Data association links observations; a filter predicts motion then corrects using measurements. Position (p_t), velocity $v_t\approx(p_t-p_{t-1})/\Delta t$. | IDs, confidence, and lost tracks are part of state estimation. Bounding boxes do not reveal full state. | What if the camera moves? |
| Optical flow: estimate each visible patch's displacement | Brightness constancy gives (I_xu+I_yv+I_t=0); nearby pixels constrain (u,v). `cv2.calcOpticalFlowPyrLK(prev_gray,next_gray,points,None)` | Measures image motion; camera motion and depth also affect it. It is not metric object velocity. | Why can one edge leave motion ambiguous? |
| Dense flow: a field of apparent movement | `cv2.calcOpticalFlowFarneback(prev_gray,next_gray,None,.5,3,15,3,5,1.2,0)` | Can aid predictive encoders, but lighting change violates brightness constancy. | What happens at an occlusion boundary? |

**Worked motion example:** a centroid moves from ((20,48)) to ((22,48)) in one frame at 30 FPS. Velocity is ((60,0)) pixels/second. Metric velocity needs camera geometry and depth. Experiment 01 computes centroids and sparse LK flow independently. Its known-color mask is a baseline with predictable failures under color shift and occlusion.

For implementation definitions consult the [OpenCV tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html). The examples here are educational constructions.

## ELI15: recognition tasks are different outputs

**Classification** answers what category a whole image belongs to. A classifier head computes class probabilities, often $p=\operatorname{softmax}(Wz)$. A glass classifier produces no location.

**Object detection** returns objects and boxes, typically $(x_{\min},y_{\min},x_{max},y_{max})$, plus confidence and class. Bounding boxes approximate extent, not collision geometry.

**Semantic segmentation** assigns a class to each pixel. Two glasses can share the same “glass” class. **Instance segmentation** distinguishes the two instances. These are valuable for object-centered states but require temporal association to preserve identity across frames.

Minimal heads, assuming an encoder already supplies `features`:

```python
classifier = torch.nn.Linear(128, 10)  # features: B,128
boxes = torch.nn.Linear(128, 4)        # educational single-box head, not a detector
segmenter = torch.nn.Conv2d(32, 5, 1) # features: B,32,H,W -> B,5,H,W
```

Real detectors need matching, multiple-object outputs, localization losses, and postprocessing. These snippets illustrate output types; they are not pretrained recognition systems.

**Check:** Which output helps estimate that a glass overhangs a table edge? What uncertainty remains?

## ELI18: cameras turn 3D into 2D

A pinhole camera projects a camera-frame point ((X,Y,Z)) to pixels:

$$
u=f_xX/Z+c_x,\qquad v=f_yY/Z+c_y.
$$

(f_x,f_y) are focal lengths in pixels; (c_x,c_y) locate the principal point. Perspective division loses depth: a small nearby object and a large distant object can occupy the same pixels.

**Intrinsics** (K) describe this projection. **Extrinsics** (R,t) map world coordinates to camera coordinates: (P_c=RP_w+t). **Calibration** estimates these parameters, often with known checkerboard corners. Lens distortion bends the ideal projection and must be modeled.

```python
K = np.array([[500,0,320],[0,500,240],[0,0,1]],dtype=float)
point_camera = np.array([.1,.2,2.0])
q = K @ point_camera
pixel = q[:2] / q[2]       # (345,290)
# With actual corresponding measurements:
# ok,K,dist,rvecs,tvecs = cv2.calibrateCamera(object_points,image_points,(640,480),None,None)
```

Coordinate frames are conventions: world, robot base, gripper, camera, and image have different axes and units. A homogeneous transform packages rotation and translation into a 4×4 matrix. Multiplication order matters. Rotations are not arbitrary matrices; they must preserve lengths and orientation.

**Check:** Why cannot pixel ((345,290)) uniquely determine the original 3D point?

## Depth, stereo, and backprojection

**Depth estimation** assigns distance along a camera ray. Learned monocular depth can exploit visual priors but may lack metric scale without calibration or training constraints.

**Stereo** compares two calibrated views. In a rectified setup, disparity (d=u_L-u_R) gives (Z=fB/d), where (B) is camera separation. With (f=500) pixels, (B=.1) meters, and (d=25) pixels, (Z=2) meters. Small disparity errors become large depth errors far away.

```python
# left_gray, right_gray must be actual calibrated, rectified paired images
# disparity = cv2.StereoSGBM_create(numDisparities=64,blockSize=5).compute(left_gray,right_gray)/16.
u,v,Z = 345,290,2.
point = np.array([(u-320)*Z/500,(v-240)*Z/500,Z])
```

Backprojection converts each depth pixel to 3D. A **point cloud** is a set of points, often with color. A **mesh** adds faces between vertices; a **voxel** representation divides volume into cells. Points may miss surfaces; meshes may contain holes; voxels cost memory cubic in resolution. None automatically supplies mass or friction.

**Check:** Why can a collision mesh derived from visual geometry still simulate contacts incorrectly?

## ELI21: motion of the camera

**Visual odometry** estimates successive camera poses. Feature matching plus geometric constraints can infer relative rotation/translation. Monocular translation generally has unresolved scale. A pose estimate is distinct from tracking a moving object.

**SLAM**, simultaneous localization and mapping, estimates camera/robot pose and a map together. A loop closure recognizes a revisited place and reduces accumulated drift. Moving objects violate a static-scene assumption; a world model may need separate dynamic object states.

Conceptual estimation objective:

$$
\min_{{R_t,t_t,P_j}}\sum_{t,j}
|\pi(K(R_tP_j+t_t))-u_{tj}|^2.
$$

(P_j) is a landmark; (u_{tj}) its measured pixel; $\pi$ applies perspective division. This is reprojection-error minimization, not a learned neural dynamics loss.

OpenCV primitives include `findEssentialMat`, `recoverPose`, `triangulatePoints`, and `solvePnP`. A functioning SLAM system additionally needs matching, outlier rejection, map management, and optimization. No five-line example can responsibly claim to implement it.

**Check:** How can camera movement create flow even when every object is stationary?

## ELI24: CNNs and Vision Transformers

A **CNN** learns local convolution filters, with shared weights at different positions. Layered receptive fields build larger context. A **Vision Transformer** breaks an image into patches, projects them into vectors, adds positional information, and mixes them using attention.

```python
cnn = torch.nn.Sequential(torch.nn.Conv2d(3,16,3,padding=1),torch.nn.ReLU())
patchify = torch.nn.Conv2d(3,64,kernel_size=8,stride=8)
tokens = patchify(torch.rand(2,3,64,64)).flatten(2).transpose(1,2) # 2,64,64
attention = torch.nn.MultiheadAttention(64,4,batch_first=True)
mixed,_ = attention(tokens,tokens,tokens)
```

Attention computes $\operatorname{softmax}(QK^\top/\sqrt d)V$: compatible queries and keys determine how values mix. ViTs are learned representations, not explicit camera geometry. Dense patch features can retain object layout better than one pooled vector, but geometry still requires evaluation. [ViT paper](https://arxiv.org/abs/2010.11929).

## ELI27: choose the weakest sufficient perception system

For a colored ball in a fixed scene, a mask and centroid are useful baselines. For changing viewpoints, occlusion, and unseen objects, learned features or geometric estimation become necessary. Ask whether errors arise from sensing, representation aliasing, dynamics, planning, or control.

A learned encoder can benefit from classical depth or flow, while classical tracking can use learned features. The categories overlap. Evaluate both against identical downstream goals rather than assuming learned methods win.

**Check:** Propose an ablation that distinguishes a better visual encoder from a better dynamics predictor.
