import React, { useState, useEffect } from "react";
import PointCloudModelViewer from "../PointCloudModelViewer/PointCloudModelViewer";
import styles from "./PCMVContainer.module.css";
type ForestName = "Karawatha" | "QCAT" | "Samford" | "Venman";

const PCMVContainer: React.FC<PCMVContainerProps> = ({
  title1,
  title2,
  file,
  isSingleViewer,
}) => {
  const [aerialFile, setAerialFile] = useState<string>(
    "karawatha_submap_aerial_2.pcd"
  );
  const [groundFile, setGroundFile] = useState<string>(
    "karawatha_submap_gnd_2.pcd"
  );
  const [isSmallScreen, setIsSmallScreen] = useState<boolean>(
    window.innerWidth < 768
  );

  const forestFiles: Record<ForestName, { aerial: string; ground: string }> = {
    Karawatha: {
      aerial: "karawatha_submap_aerial_2.pcd",
      ground: "karawatha_submap_gnd_2.pcd",
    },
    QCAT: {
      aerial: "qcat_submap_aerial_1.pcd",
      ground: "qcat_submap_gnd_1.pcd",
    },
    Samford: {
      aerial: "samford_submap_aerial_1.pcd",
      ground: "samford_submap_gnd_1.pcd",
    },
    Venman: {
      aerial: "venman_submap_aerial_1.pcd",
      ground: "venman_submap_gnd_1.pcd",
    },
  };

  const handleForestChange = (forestName: ForestName) => {
    setAerialFile(forestFiles[forestName].aerial);
    setGroundFile(forestFiles[forestName].ground);
  };

  useEffect(() => {
    const handleResize = () => setIsSmallScreen(window.innerWidth < 768);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div
      className={
        isSingleViewer
          ? styles.singleViewerContainer
          : styles.doubleViewerContainer
      }
    >
      {!isSingleViewer && (
        <div className={styles.selector}>
          <label htmlFor="forestSelector">Choose Environment:</label>
          <select
            id="forestSelector"
            onChange={(e) => handleForestChange(e.target.value as ForestName)}
            className={styles.select}
          >
            {Object.keys(forestFiles).map((forest) => (
              <option key={forest} value={forest}>
                {forest}
              </option>
            ))}
          </select>
        </div>
      )}

      {isSingleViewer && file && (
        <div className={styles.singleViewer}>
          <PointCloudModelViewer file={file} />
        </div>
      )}

      {!isSingleViewer && (
        <div
          className={styles.modelViewers}
          style={{
            flexDirection: isSmallScreen ? "column" : "row",
          }}
        >
          <div className={styles.viewerContainer}>
            <h2 className={styles.viewerTitle}>{title1}</h2>
            <PointCloudModelViewer file={aerialFile} />
          </div>
          {title2 && (
            <div className={styles.viewerContainer}>
              <h2 className={styles.viewerTitle}>{title2}</h2>
              <PointCloudModelViewer file={groundFile} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PCMVContainer;
