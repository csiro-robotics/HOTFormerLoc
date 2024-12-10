import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { PCDLoader } from "three/addons/loaders/PCDLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PointCloudModelViewerProps } from "../../types/PointCloudTypes";
import { FaSyncAlt, FaExpand } from "react-icons/fa";

import styles from "./PointCloudModelViewer.module.css";

const PointCloudModelViewer: React.FC<PointCloudModelViewerProps> = ({
  file,
  pointSize = 0.1,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const scene = new THREE.Scene();

    // Camera
    const camera = new THREE.PerspectiveCamera(
      75,
      container.offsetWidth / container.offsetHeight,
      0.01,
      10000
    );
    camera.position.set(0, 0, 200); // Start farther back for easier navigation
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.offsetWidth, container.offsetHeight);
    renderer.setClearColor(0x000000, 1); // Black background
    container.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.2;      // Smoother damping
    controls.rotateSpeed = 0.5;        // Slow down rotation speed
    controls.zoomSpeed = 1.2;          // Improve zooming ease
    controls.minDistance = 0.5;        // Allow very close zoom
    controls.maxDistance = 1000;       // Allow far zoom out
    controls.maxPolarAngle = Math.PI;  // Allow full vertical rotation
    controlsRef.current = controls;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 2);
    directionalLight.position.set(10, 10, 10).normalize();
    scene.add(directionalLight);

    // Load Point Cloud
    const loader = new PCDLoader();
    const filePath = `assets/pcd/${file}`;

    loader.load(
      filePath,
      (points) => {
        points.material.size = pointSize;
        scene.add(points);
        adjustToBoundingBox(scene, camera, controls);
      },
      undefined,
      (error) => {
        console.error("Error loading PCD file:", error);
      }
    );

    // Adjust to Bounding Box
    const adjustToBoundingBox = (
      scene: THREE.Scene,
      camera: THREE.PerspectiveCamera,
      controls: OrbitControls
    ) => {
      const boundingBox = new THREE.Box3().setFromObject(scene);
      const center = new THREE.Vector3();
      boundingBox.getCenter(center);

      const size = new THREE.Vector3();
      boundingBox.getSize(size);

      const maxDim = Math.max(size.x, size.y, size.z);
      const distance = maxDim * 3;

      camera.position.set(center.x, center.y, center.z + distance);
      camera.lookAt(center);
      controls.target.copy(center);
      controls.update();
    };

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      camera.aspect = container.offsetWidth / container.offsetHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.offsetWidth, container.offsetHeight);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      container.removeChild(renderer.domElement);
      renderer.dispose();
      window.removeEventListener("resize", handleResize);
    };
  }, [file, pointSize]);

  // Handlers
  const handleResetCamera = () => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(0, 0, 200); // Reset farther back
      cameraRef.current.lookAt(0, 0, 0);
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();
    }
  };

  const handleToggleFullscreen = () => {
    if (containerRef.current) {
      if (!document.fullscreenElement) {
        containerRef.current.requestFullscreen();
      } else {
        document.exitFullscreen();
      }
    }
  };

  return (
    <div ref={containerRef} className={styles.modelViewer}>
      <div className={styles.iconContainer}>
        <button
          onClick={handleResetCamera}
          className={styles.iconButton}
          title="Reset Camera"
        >
          <FaSyncAlt />
        </button>
        <button
          onClick={handleToggleFullscreen}
          className={styles.iconButton}
          title="Toggle Fullscreen"
        >
          <FaExpand />
        </button>
      </div>
    </div>
  );
};

export default PointCloudModelViewer;
