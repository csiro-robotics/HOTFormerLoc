import React from "react";
import styles from "./ImageWithCaption.module.css";

const ImageWithCaption: React.FC<ImageWithCaptionProps> = ({
  src,
  alt,
  caption,
  height,
  width
}) => (
  <figure className={styles.figure}>
    <img src={src} alt={alt} height={height} width={width} className={styles.image} />
    <figcaption>{caption}</figcaption>
  </figure>
);

export default ImageWithCaption;
