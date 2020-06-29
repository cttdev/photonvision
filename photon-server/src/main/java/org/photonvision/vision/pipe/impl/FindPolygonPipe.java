package org.photonvision.vision.pipe.impl;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.photonvision.vision.opencv.CVShape;
import org.photonvision.vision.opencv.Contour;
import org.photonvision.vision.opencv.ContourShape;
import org.photonvision.vision.pipe.CVPipe;

public class FindPolygonPipe
        extends CVPipe<List<Contour>, List<CVShape>, FindPolygonPipe.FindPolygonPipeParams> {
    private int corners;
    private MatOfPoint2f approx = new MatOfPoint2f();

    /*
    * Runs the process for the pipe.
    *
    * @param in Input for pipe processing.
    * @return Result of processing.
    */
    @Override
    protected List<CVShape> process(List<Contour> in) {
        // List containing all the output shapes
        List<CVShape> output = new ArrayList<>();

        for (Contour contour : in) output.add(getShape(contour));

        return output;
    }

    private CVShape getShape(Contour in) {

        corners = getCorners(in);

        /*The contourShape enum has predefined shapes for Circles, Triangles, and Quads
        meaning any shape not fitting in those predefined shapes must be a custom shape.
        */
        if (ContourShape.fromSides(corners) == null) {
            return new CVShape(in, ContourShape.Custom);
        }
        switch (ContourShape.fromSides(corners)) {
            case Circle:
                return new CVShape(in, ContourShape.Circle);
            case Triangle:
                return new CVShape(in, ContourShape.Triangle);
            case Quadrilateral:
                return new CVShape(in, ContourShape.Quadrilateral);
        }

        return new CVShape(in, ContourShape.Custom);
    }

    private int getCorners(Contour contour) {
        // Release previous approx
        approx.release();
        Imgproc.approxPolyDP(
                contour.getMat2f(),
                approx,
                // Converts an accuracy percentage between 1-100 to an epsilon
                params.accuracyPercentage / 600.0 * Imgproc.arcLength(contour.getMat2f(), true),
                true);
        // The height of the resultant approximation is the number of vertices
        return (int) approx.size().height;
    }

    public static class FindPolygonPipeParams {
        // Should be a value between 0-100
        double accuracyPercentage;

        public FindPolygonPipeParams(double accuracyPercentage) {
            this.accuracyPercentage = accuracyPercentage;
        }
    }
}
