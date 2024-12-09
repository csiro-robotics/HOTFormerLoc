interface VideoPlayerContainerProps {
  src_1_path: string;
  src_1_title: string;
  src_2_path: string;
  src_2_title: string;
}

interface VideoPlayerProps {
  src: string;
  autoPlay?: boolean;
  controls?: boolean;
  loop?: boolean;
  muted?: boolean;
}
