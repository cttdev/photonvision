package com.chameleonvision.vision.pipeline.pipes;

import com.chameleonvision.config.CameraCalibrationConfig;
import com.chameleonvision.vision.pipeline.Pipe;
import com.chameleonvision.vision.pipeline.impl.StandardCVPipeline;
import com.chameleonvision.vision.pipeline.impl.StandardCVPipelineSettings;
import edu.wpi.first.wpilibj.geometry.Pose2d;
import edu.wpi.first.wpilibj.geometry.Rotation2d;
import edu.wpi.first.wpilibj.geometry.Translation2d;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.util.FastMath;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.*;

public class SolvePNPPipe implements Pipe<Pair<List<StandardCVPipeline.TrackedTarget>, Mat>, List<StandardCVPipeline.TrackedTarget>> {

    private Double tilt_angle;
    private MatOfPoint3f objPointsMat = new MatOfPoint3f();
    private Mat rVec = new Mat();
    private Mat tVec = new Mat();
    private Mat rodriguez = new Mat();
    private Mat pzero_world = new Mat();
    private Mat cameraMatrix = new Mat();
    Mat rot_inv = new Mat();
    Mat kMat = new Mat();
    private MatOfDouble distortionCoefficients = new MatOfDouble();
    private List<StandardCVPipeline.TrackedTarget> poseList = new ArrayList<>();
    Comparator<Point> leftRightComparator = Comparator.comparingDouble(point -> point.x);
    Comparator<Point> verticalComparator = Comparator.comparingDouble(point -> point.y);
    private double distanceDivisor = 1.0;
    Mat scaledTvec = new Mat();
    MatOfPoint2f boundingBoxResultMat = new MatOfPoint2f();
    MatOfPoint2f polyOutput = new MatOfPoint2f();
    private Mat greyImg = new Mat();

    public SolvePNPPipe(StandardCVPipelineSettings settings, CameraCalibrationConfig calibration, Rotation2d tilt) {
        super();
        setCameraCoeffs(calibration);
//        setBoundingBoxTarget(settings.targetWidth, settings.targetHeight);
        // TODO add proper year differentiation
        set2020Target(true);

        this.tilt_angle = tilt.getRadians();
    }

    public void set2020Target(boolean isHighGoal) {
        if(isHighGoal) {
            // tl, bl, br, tr is the order
            List<Point3> corners = List.of(

                new Point3(-19.625, 0, 0),
                new Point3(-9.819867, -17, 0),
                new Point3(9.819867, -17, 0),
                new Point3(19.625, 0, 0));
            setObjectCorners(corners);
        } else {
            setBoundingBoxTarget(7, 11);
        }
    }

    public void setBoundingBoxTarget(double targetWidth, double targetHeight) {
        // order is left top, left bottom, right bottom, right top

        List<Point3> corners = List.of(
                new Point3(-targetWidth / 2.0, targetHeight / 2.0, 0.0),
                new Point3(-targetWidth / 2.0, -targetHeight / 2.0, 0.0),
                new Point3(targetWidth / 2.0, -targetHeight / 2.0, 0.0),
                new Point3(targetWidth / 2.0, targetHeight / 2.0, 0.0)
        );
        setObjectCorners(corners);
    }

    public void setObjectCorners(List<Point3> objectCorners) {
        objPointsMat.release();
        objPointsMat = new MatOfPoint3f();
        objPointsMat.fromList(objectCorners);
    }

    public void setConfig(StandardCVPipelineSettings settings, CameraCalibrationConfig camConfig, Rotation2d tilt) {
        setCameraCoeffs(camConfig);
//        setBoundingBoxTarget(settings.targetWidth, settings.targetHeight);
        // TODO add proper year differentiation
        tilt_angle = tilt.getRadians();
        this.objPointsMat = settings.targetCornerMat;
    }

    private void setCameraCoeffs(CameraCalibrationConfig settings) {
        if(settings == null) {
            System.err.println("SolvePNP can only run on a calibrated resolution, and this one is not! Please calibrate to use solvePNP.");
            return;
        }
        if(cameraMatrix != settings.getCameraMatrixAsMat()) {
            cameraMatrix.release();
            settings.getCameraMatrixAsMat().copyTo(cameraMatrix);
        }
        if(distortionCoefficients != settings.getDistortionCoeffsAsMat()) {
            distortionCoefficients.release();
            settings.getDistortionCoeffsAsMat().copyTo(distortionCoefficients);
        }
        this.distanceDivisor = settings.squareSize;
    }

    @Override
    public Pair<List<StandardCVPipeline.TrackedTarget>, Long> run(Pair<List<StandardCVPipeline.TrackedTarget>, Mat> imageTargetPair) {
        long processStartNanos = System.nanoTime();
        var targets = imageTargetPair.getLeft();
        var image = imageTargetPair.getRight();
        poseList.clear();
        for(var target: targets) {
            var corners = find2020VisionTarget(target);//, imageTargetPair.getRight()); //find2020VisionTarget(target);// (target.leftRightDualTargetPair != null) ? findCorner2019(target) : findBoundingBoxCorners(target);
//            var corners = findCorner2019(target);
            if(corners == null) continue;

            // refine the estimate
            corners = refineCornerEstimateSubPix(corners, image);

            var pose = calculatePose(corners, target);
            if(pose != null) poseList.add(pose);
        }
        long processTime = System.nanoTime() - processStartNanos;
        return Pair.of(poseList, processTime);
    }

    private MatOfPoint2f findCorner2019(StandardCVPipeline.TrackedTarget target) {
        if(target.leftRightDualTargetPair == null) return null;

        var left = target.leftRightDualTargetPair.getLeft();
        var right = target.leftRightDualTargetPair.getRight();

        // flip if the "left" target is to the right
        if(left.x > right.x) {
            var temp = left;
            left = right;
            right = temp;
        }

        var points = new MatOfPoint2f();
        points.fromArray(
                new Point(left.x, left.y + left.height),
                new Point(left.x, left.y),
                new Point(right.x + right.width, right.y),
                new Point(right.x + right.width, right.y + right.height)
        );
        return points;
    }

    MatOfPoint2f target2020ResultMat = new MatOfPoint2f();

    private double distanceBetween(Point a, Point b) {
        return FastMath.sqrt(FastMath.pow(a.x - b.x, 2) + FastMath.pow(a.y - b.y, 2));
    }

    /**
     * Find the target using the outermost tape corners and a 2020 target.
     * @param target the target.
     * @return The four outermost tape corners.
     */
    private MatOfPoint2f find2020VisionTarget(StandardCVPipeline.TrackedTarget target) {
        if(target.rawContour.cols() < 1) return null;

        var centroid = target.minAreaRect.center;
        Comparator<Point> distanceProvider = Comparator.comparingDouble((Point point) -> FastMath.sqrt(FastMath.pow(centroid.x - point.x, 2) + FastMath.pow(centroid.y - point.y, 2)));

        // algorithm from team 4915

        // Contour perimeter
        var peri = Imgproc.arcLength(target.rawContour, true);
        // approximating a shape around the contours
        // Can be tuned to allow/disallow hulls
        // Approx is the number of vertices
        // Ramer–Douglas–Peucker algorithm
        Imgproc.approxPolyDP(target.rawContour, polyOutput, 0.02 * peri, true);

        var area = Imgproc.moments(polyOutput);

//        if (area.get_m00() < 200) {
//            return null;
//        }

        var polyList = polyOutput.toList();

        polyOutput.copyTo(target.approxPoly);

        // left top, left bottom, right bottom, right top
        var boundingBoxCorners = findBoundingBoxCorners(target).toList();

        try {

            // top left and top right are the poly corners closest to the bouding box tl and tr
            var tl = polyList.stream().min(Comparator.comparingDouble((Point p) -> distanceBetween(p, boundingBoxCorners.get(0)))).get();
            var tr = polyList.stream().min(Comparator.comparingDouble((Point p) -> distanceBetween(p, boundingBoxCorners.get(3)))).get();

            var bl = polyList.stream().filter(point -> point.x < centroid.x && point.y > centroid.y).max(distanceProvider).get();
            var br = polyList.stream().filter(point -> point.x > centroid.x && point.y > centroid.y).max(distanceProvider).get();

//            polyList = new ArrayList<>(polyList);
//            polyList.removeAll(List.of(tl, tr, bl, br));
//
//            var tl2 = polyList.stream().min(Comparator.comparingDouble((Point p) -> distanceBetween(p, boundingBoxCorners.get(0)))).get();
//            var tr2 = polyList.stream().min(Comparator.comparingDouble((Point p) -> distanceBetween(p, boundingBoxCorners.get(3)))).get();
//
//            var bl2 = polyList.stream().filter(point -> point.x < centroid.x && point.y > centroid.y).max(distanceProvider).get();
//            var br2 = polyList.stream().filter(point -> point.x > centroid.x && point.y > centroid.y).max(distanceProvider).get();

            target2020ResultMat.release();
            target2020ResultMat.fromList(List.of(tl, bl, br, tr));//, tr2, br2, bl2, tl2));

            return target2020ResultMat;
        } catch (NoSuchElementException e) {
            return null;
        }
    }

    /**
     * Find the target using the outermost tape corners and a dual target.
     * @param target the target.
     * @return The four outermost tape corners.
     */
    private MatOfPoint2f findDualTargetCornerMinAreaRect(StandardCVPipeline.TrackedTarget target) {
        if(target.leftRightRotatedRect == null) return null;

        var centroid = target.minAreaRect.center;
        Comparator<Point> distanceProvider = Comparator.comparingDouble((Point point) -> FastMath.sqrt(FastMath.pow(centroid.x - point.x, 2) + FastMath.pow(centroid.y - point.y, 2)));

        var left = target.leftRightRotatedRect.getLeft();
        var right = target.leftRightRotatedRect.getRight();

        // flip if the "left" target is to the right
        if(left.center.x > right.center.x) {
            var temp = left;
            left = right;
            right = temp;
        }

        var leftPoints = new Point[4];
        left.points(leftPoints);
        var rightPoints = new Point[4];
        right.points(rightPoints);
        ArrayList<Point> combinedList = new ArrayList<>(List.of(leftPoints));
        combinedList.addAll(List.of(rightPoints));

        // start looking in the top left quadrant
        var tl = combinedList.stream().filter(point -> point.x < centroid.x && point.y < centroid.y).max(distanceProvider).get();
        var tr = combinedList.stream().filter(point -> point.x > centroid.x && point.y < centroid.y).max(distanceProvider).get();
        var bl = combinedList.stream().filter(point -> point.x < centroid.x && point.y > centroid.y).max(distanceProvider).get();
        var br = combinedList.stream().filter(point -> point.x > centroid.x && point.y > centroid.y).max(distanceProvider).get();

        boundingBoxResultMat.release();
        boundingBoxResultMat.fromList(List.of(tl, bl, br, tr));

        return boundingBoxResultMat;
    }

    /**
     *
     * @param target the target to find the corners of.
     * @return the corners. left top, left bottom, right bottom, right top
     */
    private MatOfPoint2f findBoundingBoxCorners(StandardCVPipeline.TrackedTarget target) {

//        List<Pair<MatOfPoint2f, CVPipeline2d.Target2d>> list = new ArrayList<>();
//        // find the corners based on the bounding box
//        // order is left top, left bottom, right bottom, right top

        // extract the corners
        var points = new Point[4];
        target.minAreaRect.points(points);

        // find the tl/tr/bl/br corners
        // first, min by left/right
        var list_ = Arrays.asList(points);
        list_.sort(leftRightComparator);
        // of this, we now have left and right
        // sort to get top and bottom
        var left = new ArrayList<>(List.of(list_.get(0), list_.get(1)));
        left.sort(verticalComparator);
        var right = new ArrayList<>(List.of(list_.get(2), list_.get(3)));
        right.sort(verticalComparator);

        // tl tr bl br
        var tl = left.get(0);
        var bl = left.get(1);
        var tr = right.get(0);
        var br = right.get(1);

        boundingBoxResultMat.release();
        boundingBoxResultMat.fromList(List.of(tl, bl, br, tr));

        return boundingBoxResultMat;
    }

    // Set the needed parameters to find the refined corners
    Size winSize = new Size(12, 12);
    Size zeroZone = new Size(-1, -1); // we don't need a zero zone
    TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 50, 0.001);

    private boolean shouldRefineCorners = true;

    /**
     * Refine an estimated corner position using the cornerSubPixel algorithm.
     *
     * TODO should this be here or before the points are chosen? 
     *
     * @param corners the corners detected -- this mat is modified!
     * @param img the image taken by the camera as color
     * @return the updated mat, same as the corner mat passed in.
     */
    private MatOfPoint2f refineCornerEstimateSubPix(MatOfPoint2f corners, Mat img) {
        if(!shouldRefineCorners) return corners; // just return

        Imgproc.cvtColor(img, greyImg, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cornerSubPix(greyImg, corners, winSize, zeroZone, criteria);

        return corners;
    }

    private StandardCVPipeline.TrackedTarget calculatePose(MatOfPoint2f imageCornerPoints, StandardCVPipeline.TrackedTarget target) {
        if(objPointsMat.rows() != imageCornerPoints.rows() || cameraMatrix.rows() < 2 || distortionCoefficients.cols() < 4) {
            System.err.println("can't do solvePNP with invalid params!");
            return null;
        }

        imageCornerPoints.copyTo(target.imageCornerPoints);

        try {
            Calib3d.solvePnP(objPointsMat, imageCornerPoints, cameraMatrix, distortionCoefficients, rVec, tVec);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

        // Algorithm from team 5190 Green Hope Falcons

//        var tilt_angle = 0.0; // TODO add to settings

        var x = tVec.get(0, 0)[0];
        var z = FastMath.sin(tilt_angle) * tVec.get(1, 0)[0] + tVec.get(2, 0)[0] *  FastMath.cos(tilt_angle);

        // distance in the horizontal plane between camera and target
        var distance = FastMath.sqrt(x * x + z * z);

        // horizontal angle between center camera and target
        @SuppressWarnings("SuspiciousNameCombination")
        var angle1 = FastMath.atan2(x, z);

        Calib3d.Rodrigues(rVec, rodriguez);
        Core.transpose(rodriguez, rot_inv); // rodrigurz.t()

        // This should be pzero_world = numpy.matmul(rot_inv, -tvec)
//        pzero_world  = rot_inv.mul(matScale(tVec, -1));
        scaledTvec = matScale(tVec, -1);
        Core.gemm(rot_inv, scaledTvec, 1, kMat, 0, pzero_world);

        var angle2 = FastMath.atan2(pzero_world.get(0, 0)[0], pzero_world.get(2, 0)[0]);

        var targetAngle = -angle1; // radians
        var targetRotation = -angle2; // radians
        var targetDistance = distance * 25.4 / 1000d / distanceDivisor; // This should be meters

        var targetLocation = new Translation2d(targetDistance * FastMath.cos(targetAngle), targetDistance * FastMath.sin(targetAngle));
        target.cameraRelativePose = new Pose2d(targetLocation, new Rotation2d(targetRotation));
        target.rVector = rVec;
        target.tVector = tVec;

        return target;
    }

    /**
     * Element-wise scale a matrix by a given factor
     * @param src the source matrix
     * @param factor by how much to scale each element
     * @return the scaled matrix
     */
    public Mat matScale(Mat src, double factor) {
        Mat dst = new Mat(src.rows(),src.cols(),src.type());
        Scalar s = new Scalar(factor); // TODO check if we need to add more elements to this
        Core.multiply(src, s, dst);
        return dst;
    }

}
