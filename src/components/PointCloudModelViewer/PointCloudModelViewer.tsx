import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { PCDLoader } from "three/addons/loaders/PCDLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { FaSyncAlt, FaExpand } from "react-icons/fa";

import styles from "./PointCloudModelViewer.module.css";

const PointCloudModelViewer: React.FC<PointCloudModelViewerProps> = ({
  file,
  pointSize = 0.1,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const scene = new THREE.Scene();

    // Camera setup
    const aspect = container.offsetWidth / container.offsetHeight;
    const frustumSize = 100;
    const camera = new THREE.OrthographicCamera(
      (-frustumSize * aspect) / 2,
      (frustumSize * aspect) / 2,
      frustumSize / 2,
      -frustumSize / 2,
      -10000,
      10000
    );
    camera.position.set(0, 0, 200);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.offsetWidth, container.offsetHeight);
    renderer.setClearColor(0x000000, 1);
    container.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.2;
    controls.rotateSpeed = 0.5;
    controls.zoomSpeed = 1.2;
    controls.minDistance = 0.5;
    controls.maxDistance = 2000;
    controls.maxPolarAngle = Math.PI;
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

    // Adjust camera to bounding box
    const adjustToBoundingBox = (
      scene: THREE.Scene,
      camera: THREE.OrthographicCamera,
      controls: OrbitControls
    ) => {
      const boundingBox = new THREE.Box3().setFromObject(scene);
      const center = new THREE.Vector3();
      boundingBox.getCenter(center);

      const size = new THREE.Vector3();
      boundingBox.getSize(size);

      const maxDim = Math.max(size.x, size.y, size.z);
      const distance = maxDim * 1.5;

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
      const aspect = container.offsetWidth / container.offsetHeight;
      camera.left = (-frustumSize * aspect) / 2;
      camera.right = (frustumSize * aspect) / 2;
      camera.top = frustumSize / 2;
      camera.bottom = -frustumSize / 2;
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
      const camera = cameraRef.current;
      const controls = controlsRef.current;
      camera.position.set(0, 0, 200);
      camera.zoom = 1;
      camera.updateProjectionMatrix();
      controls.target.set(0, 0, 0);
      controls.update();
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
