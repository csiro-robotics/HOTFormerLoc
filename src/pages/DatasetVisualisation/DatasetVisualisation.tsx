import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { PCDLoader } from "three/addons/loaders/PCDLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import styles from "./DatasetVisualisation.module.css";

type ForestName = "Karawatha" | "QCAT" | "Samford" | "Venman";

const DatasetVisualisation = () => {
  const aerialRef = useRef<HTMLDivElement | null>(null);
  const groundRef = useRef<HTMLDivElement | null>(null);

  const [aerialFile, setAerialFile] = useState<string>("karawatha_submap_aerial_1.pcd");
  const [groundFile, setGroundFile] = useState<string>("karawatha_submap_gnd_1.pcd");
  const [isSmallScreen, setIsSmallScreen] = useState<boolean>(window.innerWidth < 768);

  const forestFiles: Record<ForestName, { aerial: string; ground: string }> = {
    Karawatha: { aerial: "karawatha_submap_aerial_2.pcd", ground: "karawatha_submap_gnd_2.pcd" },
    QCAT: { aerial: "qcat_submap_aerial_1.pcd", ground: "qcat_submap_gnd_1.pcd" },
    Samford: { aerial: "samford_submap_aerial_1.pcd", ground: "samford_submap_gnd_1.pcd" },
    Venman: { aerial: "venman_submap_aerial_1.pcd", ground: "venman_submap_gnd_1.pcd" },
  };

  const setupScene = (
    container: HTMLDivElement | null,
    file: string,
    adjustToBoundingBox: (scene: THREE.Scene, camera: THREE.PerspectiveCamera, controls: OrbitControls) => void
  ) => {
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.offsetWidth / container.offsetHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    const controls = new OrbitControls(camera, renderer.domElement);

    renderer.setSize(container.offsetWidth, container.offsetHeight);
    container.appendChild(renderer.domElement);

    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.minDistance = 10;
    controls.maxDistance = 500;
    controls.maxPolarAngle = Math.PI / 2;
    controls.minPolarAngle = 0;

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(10, 10, 10).normalize();
    scene.add(light);

    const loader = new PCDLoader();
    loader.load(
      `assets/pcd/${file}`,
      (points) => {
        scene.add(points);
        adjustToBoundingBox(scene, camera, controls);
      },
      undefined,
      (error) => {
        console.error("Error loading PCD file:", error);
      }
    );

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const resetCamera = () => {
      adjustToBoundingBox(scene, camera, controls);
    };

    const handleResize = () => {
      camera.aspect = container.offsetWidth / container.offsetHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.offsetWidth, container.offsetHeight);
    };
    window.addEventListener("resize", handleResize);

    return {
      cleanup: () => {
        container.removeChild(renderer.domElement);
        renderer.dispose();
        window.removeEventListener("resize", handleResize);
      },
      resetCamera,
    };
  };

  const adjustToBoundingBox = (scene: THREE.Scene, camera: THREE.PerspectiveCamera, controls: OrbitControls) => {
    const boundingBox = new THREE.Box3().setFromObject(scene);
    const center = new THREE.Vector3();
    boundingBox.getCenter(center);

    const size = new THREE.Vector3();
    boundingBox.getSize(size);

    const maxDim = Math.max(size.x, size.y, size.z);
    const distance = maxDim * 2;

    camera.position.set(center.x, center.y, center.z + distance);
    controls.target.copy(center);
    controls.update();
  };

  useEffect(() => {
    const aerialScene = setupScene(aerialRef.current, aerialFile, adjustToBoundingBox);
    const groundScene = setupScene(groundRef.current, groundFile, adjustToBoundingBox);

    const handleWindowResize = () => {
      setIsSmallScreen(window.innerWidth < 768);
    };
    window.addEventListener("resize", handleWindowResize);

    return () => {
      if (aerialScene) aerialScene.cleanup();
      if (groundScene) groundScene.cleanup();
      window.removeEventListener("resize", handleWindowResize);
    };
  }, [aerialFile, groundFile]);

  const handleForestChange = (forestName: ForestName) => {
    setAerialFile(forestFiles[forestName].aerial);
    setGroundFile(forestFiles[forestName].ground);
  };

  if (isSmallScreen) {
    return (
      <div className={styles.container}>
        <header className={styles.header}>
          <h1>Dataset Visualisation</h1>
          <p>Explore aerial and ground point cloud datasets for different forests.</p>
        </header>
        <p className={styles.warning}>
          Please use a larger device or increase your window size for the best experience.
        </p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>Dataset Visualisation</h1>
        <p>Explore aerial and ground point cloud datasets for different forests.</p>
      </header>
      <div className={styles.selector}>
        <label htmlFor="forestSelector">Choose Forest:</label>
        <select
          id="forestSelector"
          onChange={(e) => handleForestChange(e.target.value as ForestName)}
          className={styles.select}
        >
          {Object.keys(forestFiles).map((forest) => (
            <option key={forest} value={forest}>
              {forest}
            </option>
          ))}
        </select>
      </div>
      <div className={styles.modelViewers}>
        <div className={styles.viewerContainer}>
          <div ref={aerialRef} className={styles.modelViewer}></div>
        </div>
        <div className={styles.viewerContainer}>
          <div ref={groundRef} className={styles.modelViewer}></div>
        </div>
      </div>
    </div>
  );
};

export default DatasetVisualisation;
