import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { PCDLoader } from "three/addons/loaders/PCDLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import styles from "./PointCloudModelViewer.module.css";

interface PointCloudModelViewerProps {
  file: string;
}

const PointCloudModelViewer: React.FC<PointCloudModelViewerProps> = ({ file }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(
      75,
      container.offsetWidth / container.offsetHeight,
      0.01,  // Adjust near clipping plane
      10000  // Adjust far clipping plane
    );
    camera.position.set(0, 0, 100);  // Initial camera position

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.offsetWidth, container.offsetHeight);
    renderer.setClearColor(0x000000, 1); // Black background
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.minDistance = 10;
    controls.maxDistance = 500;
    controls.maxPolarAngle = Math.PI / 2;
    controls.minPolarAngle = 0;

    // Add Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 2);
    directionalLight.position.set(10, 10, 10).normalize();
    scene.add(directionalLight);

    const loader = new PCDLoader();
    const filePath = `assets/pcd/${file}`;

    loader.load(
      filePath,
      (points) => {
        console.log("Loaded points:", points);
        scene.add(points);
        adjustToBoundingBox(scene, camera, controls);
      },
      undefined,
      (error) => {
        console.error("Error loading PCD file:", error);
      }
    );

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
      const distance = maxDim * 2;

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
  }, [file]);

  return <div ref={containerRef} className={styles.modelViewer}></div>;
};

export default PointCloudModelViewer;
