interface ContentBlockProps {
    imageSrc: string;
    altText: string;
    caption: string;
    description: string;
    height?: string;
    width?: string;
  }

  interface CheckpointRowProps {
    model_name: string;
    dataset: string;
    weights_name: string;
    link: string;
  }