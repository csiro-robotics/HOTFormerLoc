import React from "react";
import VideoPlayer from "../VideoPlayer/VideoPlayer";
import styles from "./VideoPlayerContainer.module.css";

const VideoPlayerContainer: React.FC<VideoPlayerContainerProps> = ({
  src_1_title,
  src_1_path,
  src_2_title,
  src_2_path,
}) => {
  return (
    <div className={styles.videoPlayerContainer}>
      <div className={styles.videoWrapper}>
        <h2 className={styles.videoTitle}>{src_1_title}</h2>
        <VideoPlayer src={src_1_path} />
      </div>

      <div className={styles.videoWrapper}>
        <h2 className={styles.videoTitle}>{src_2_title}</h2>
        <VideoPlayer src={src_2_path} />
      </div>
    </div>
  );
};

export default VideoPlayerContainer;
