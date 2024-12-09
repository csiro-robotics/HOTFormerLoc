import React from "react";
import styles from "./ImageWithCaption.module.css";
const ImageWithCaption: React.FC<ImageWithCaptionProps> = ({
  src,
  alt,
  caption,
}) => (
  <figure className={styles.figure}>
    <img src={src} alt={alt} className={styles.image} />
    <figcaption>{caption}</figcaption>
  </figure>
);

export default ImageWithCaption;
