
import React from "react";
import ImageWithCaption from "../ImageWithCaption/ImageWithCaption";
import styles from "./ContentBlock.module.css"

const ContentBlock: React.FC<ContentBlockProps> = ({
    imageSrc,
    altText,
    caption,
    description,
    height, 
    width
  }) => (
    <div className={styles.imageGrid}>
      <ImageWithCaption src={imageSrc} alt={altText} caption={caption} height={height} width={width} />
      <p className={styles.paragraph}>{description}</p>
    </div>
  );

export default ContentBlock;