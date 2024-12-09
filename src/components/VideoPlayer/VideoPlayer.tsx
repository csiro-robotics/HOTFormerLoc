import React from "react";
import styles from "./VideoPlayer.module.css";

interface VideoPlayerProps {
  src: string;
  autoPlay?: boolean;
  controls?: boolean;
  loop?: boolean;
  muted?: boolean;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({
  src,
  autoPlay = true,
  controls = true,
  loop = true,
  muted = true,
}) => {
  return (
    <div className={styles.container}>
      <video
        className={styles.video}
        src={src}
        autoPlay={autoPlay}
        controls={controls}
        loop={loop}
        muted={muted}
      >
        Your browser does not support the video tag.
      </video>
    </div>
  );
};

export default VideoPlayer;
