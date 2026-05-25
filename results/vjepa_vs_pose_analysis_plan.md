# Analysis Plan: Why V-JEPA Outperforms Pose-Only

Goal: understand whether V-JEPA improves fall anticipation by using non-pose visual cues such as scene context, furniture, floor layout, occlusion, and appearance/motion information that is not captured by RTMPose keypoints.

## 1. Error Overlap Analysis

Compare test-set predictions from:

- Pose-only Transformer
- V-JEPA predictive model
- Pose + V-JEPA fusion model

Create four groups:

- V-JEPA correct, pose wrong
- Pose correct, V-JEPA wrong
- Both correct
- Both wrong

For representative samples, show:

- Observed input clip frames
- Ground-truth fall onset
- Pose probability
- V-JEPA probability
- Fusion probability
- Ground-truth label

This directly identifies what types of examples V-JEPA handles better than pose.

## 2. Background and Context Categorization

Manually inspect examples where V-JEPA is correct and pose is wrong. Assign each example to one or more categories:

- Furniture, bed, chair, or object context nearby
- Visible floor / scene layout cues
- Camera angle or viewpoint
- Person partially occluded
- Person partially outside frame
- Pre-fall object interaction
- Poor lighting or clutter
- Pose detector failure
- Multiple people or distracting objects

Report the frequency of each category among V-JEPA-correct / pose-wrong examples.

## 3. Pose Quality vs. Prediction Error

For each window, compute pose-quality features:

- Mean keypoint confidence
- Fraction of low-confidence keypoints
- Torso keypoint confidence
- Fraction of missing joints
- Pose velocity magnitude

Compare these distributions across:

- Pose correct vs. pose wrong
- V-JEPA correct vs. V-JEPA wrong
- Fusion correct vs. fusion wrong

Hypothesis: pose-only errors should correlate with low pose confidence or noisy keypoints, while V-JEPA may be more robust because it uses full visual context.

## 4. Pose Skeleton Overlay Examples

For selected test examples, overlay RTMPose keypoints on the observed frames.

Good examples to include:

- Pose model false negative, V-JEPA true positive
- Pose model false positive, V-JEPA true negative
- Fusion improves over both single-modality models

This helps visually explain whether pose failures are caused by occlusion, truncation, bad keypoints, or ambiguous body motion.

## 5. V-JEPA Nearest-Neighbor Retrieval

Use pooled observed V-JEPA latents to retrieve nearest training windows for selected test windows.

Procedure:

- Average observed V-JEPA latent tokens over time.
- Compute cosine similarity between test and train windows.
- Display the top-k nearest training clips.

Research question:

- Does V-JEPA cluster visually similar pre-fall contexts?
- Are fall-risk clips close to other fall-risk clips even when the pose pattern differs?

## 6. Background Ablation

Run V-JEPA or fusion inference on modified versions of clips:

- Original clip
- Person-cropped clip
- Background-blurred clip
- Person-masked / background-only clip

Compare fall probabilities across these versions.

Interpretation:

- If probability drops after background removal, scene context likely matters.
- If probability stays high with only the person crop, V-JEPA is mostly using body appearance/motion.
- If background-only still has signal, the dataset may contain scene/context bias.

Simpler first version: crop around the pose bounding box with margin and compare original vs. person crop.

## 7. Temporal Probability Curves

For selected raw videos, slide the observation window over time and plot:

- x-axis: time before fall onset
- y-axis: predicted fall probability
- lines: pose-only, V-JEPA, fusion

This can show whether:

- V-JEPA probability rises earlier than pose-only
- Pose-only only reacts once body motion becomes obvious
- Fusion balances early context and pose dynamics

## 8. Occlusion Sensitivity

For V-JEPA or fusion, occlude patches of input frames and measure the change in predicted fall probability.

Output:

- A rough heatmap over the frame showing which regions most affect prediction.

This is more practical than Grad-CAM for V-JEPA and can support the claim that non-pose regions contribute to predictions.

## Recommended Priority

Start with:

1. Error overlap analysis
2. Pose quality vs. prediction error
3. Background/person-crop ablation

These three analyses give the clearest research story:

> V-JEPA outperforms pose-only because pose features can be noisy or incomplete, while V-JEPA captures body appearance, motion, objects, scene layout, and other visual context useful for anticipating falls.
