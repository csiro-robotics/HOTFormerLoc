
import React from "react";
import ImageWithCaption from "../ImageWithCaption/ImageWithCaption";
import styles from "./ContentBlock.module.css"

const ContentBlock: React.FC<ContentBlockProps> = ({
    imageSrc,
    altText,
    caption,
    description,
  }) => (
    <div className={styles.imageGrid}>
      <ImageWithCaption src={imageSrc} alt={altText} caption={caption} />
      <p className={styles.paragraph}>{description}</p>
    </div>
  );

export default ContentBlock;