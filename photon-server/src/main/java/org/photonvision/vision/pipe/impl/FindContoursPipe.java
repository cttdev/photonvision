package org.photonvision.vision.pipe.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.imgproc.Imgproc;
import org.photonvision.vision.opencv.Contour;
import org.photonvision.vision.pipe.CVPipe;

public class FindContoursPipe
        extends CVPipe<Mat, List<Contour>, FindContoursPipe.FindContoursParams> {

    private List<MatOfPoint> m_foundContours = new ArrayList<>();

    @Override
    protected List<Contour> process(Mat in) {
        for (var m : m_foundContours) {
            m.release(); // necessary?
        }
        m_foundContours.clear();

        Imgproc.findContours(
                in, m_foundContours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_TC89_L1);

        return m_foundContours.stream().map(Contour::new).collect(Collectors.toList());
    }

    public static class FindContoursParams {}
}
