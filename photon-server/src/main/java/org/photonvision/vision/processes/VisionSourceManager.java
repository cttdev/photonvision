package org.photonvision.vision.processes;

import edu.wpi.cscore.UsbCamera;
import edu.wpi.cscore.UsbCameraInfo;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.StringUtils;
import org.photonvision.common.configuration.CameraConfiguration;
import org.photonvision.vision.camera.CameraType;
import org.photonvision.vision.camera.USBCameraSource;
import org.photonvision.vision.frame.provider.NetworkFrameProvider;

public class VisionSourceManager {
    public List<VisionSource> LoadAllSources(List<CameraConfiguration> camerasConfiguration) {
        return LoadAllSources(camerasConfiguration, Arrays.asList(UsbCamera.enumerateUsbCameras()));
    }

    public List<VisionSource> LoadAllSources(
            List<CameraConfiguration> camerasConfiguration, List<UsbCameraInfo> usbCameraInfos) {
        var UsbCamerasConfiguration =
                camerasConfiguration.stream()
                        .filter(configuration -> configuration.cameraType == CameraType.UsbCamera)
                        .collect(Collectors.toList());
        // var HttpCamerasConfiguration = camerasConfiguration.stream().filter(configuration ->
        // configuration.cameraType == CameraType.HttpCamera);
        var matchedCameras = matchUSBCameras(usbCameraInfos, UsbCamerasConfiguration);
        return loadUSBCameraSources(matchedCameras);
    }

    private NetworkFrameProvider loadHTTPCamera(CameraConfiguration config) {
        throw new NotImplementedException("");
    }

    private List<CameraConfiguration> matchUSBCameras(
            List<UsbCameraInfo> infos, List<CameraConfiguration> cameraConfigurationList) {
        ArrayList<UsbCameraInfo> loopableInfo = new ArrayList<>(infos);
        List<CameraConfiguration> cameraConfigurations = new ArrayList<>();

        for (CameraConfiguration config : cameraConfigurationList) {
            UsbCameraInfo cameraInfo;
            if (!StringUtils.isNumeric(config.path)) {
                // matching by path
                cameraInfo =
                        loopableInfo.stream()
                                .filter(usbCameraInfo -> usbCameraInfo.path.equals(config.path))
                                .findFirst()
                                .orElse(null);
            } else {
                // match by index
                cameraInfo =
                        loopableInfo.stream()
                                .filter(usbCameraInfo -> usbCameraInfo.dev == Integer.parseInt(config.path))
                                .findFirst()
                                .orElse(null);
            }

            if (cameraInfo != null) {
                loopableInfo.remove(cameraInfo);
                cameraConfigurations.add(config);
            }
        }
        for (UsbCameraInfo info : loopableInfo) {
            // create new camera config for all new cameras
            String name = info.name.replaceAll("[^\\x00-\\x7F]", "");
            String uniqueName = name;
            int suffix = 0;

            while (containsName(cameraConfigurations, uniqueName)) {
                suffix++;
                uniqueName = String.format("%s (%d)", uniqueName, suffix);
            }

            CameraConfiguration configuration =
                    new CameraConfiguration(name, uniqueName, uniqueName, ((Integer) info.dev).toString());
            cameraConfigurations.add(configuration);
        }

        return cameraConfigurations;
    }

    private List<VisionSource> loadUSBCameraSources(List<CameraConfiguration> configurations) {
        List<VisionSource> usbCameraSources = new ArrayList<>();
        configurations.forEach(
                configuration -> usbCameraSources.add(new USBCameraSource(configuration)));
        return usbCameraSources;
    }

    private boolean containsName(final List<CameraConfiguration> list, final String name) {
        return list.stream().anyMatch(configuration -> configuration.uniqueName.equals(name));
    }
}
