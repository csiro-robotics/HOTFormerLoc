import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { PCDLoader } from "three/addons/loaders/PCDLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

type ForestName = "Karawatha" | "QCAT" | "Samford" | "Venman";

const DatasetVisualisation = () => {
  const aerialRef = useRef<HTMLDivElement | null>(null);
  const groundRef = useRef<HTMLDivElement | null>(null);

  const [aerialFile, setAerialFile] = useState<string>("karawatha_submap_aerial_1.pcd");
  const [groundFile, setGroundFile] = useState<string>("karawatha_submap_gnd_1.pcd");

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
    const renderer = new THREE.WebGLRenderer();
    const controls = new OrbitControls(camera, renderer.domElement);

    renderer.setSize(container.offsetWidth, container.offsetHeight);
    container.appendChild(renderer.domElement);

    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 10;
    controls.maxDistance = 500;

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

    window.addEventListener("resize", () => {
      camera.aspect = container.offsetWidth / container.offsetHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.offsetWidth, container.offsetHeight);
    });

    return () => {
      container.removeChild(renderer.domElement);
      renderer.dispose();
      window.removeEventListener("resize", () => {});
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
    const cleanupAerial = setupScene(aerialRef.current, aerialFile, adjustToBoundingBox);
    const cleanupGround = setupScene(groundRef.current, groundFile, adjustToBoundingBox);

    return () => {
      if (cleanupAerial) cleanupAerial();
      if (cleanupGround) cleanupGround();
    };
  }, [aerialFile, groundFile]);

  const handleForestChange = (forestName: ForestName) => {
    setAerialFile(forestFiles[forestName].aerial);
    setGroundFile(forestFiles[forestName].ground);
  };

  return (
    <div>
      <div id="overview">
        <h3>
          Overview
        </h3>
        <p>Some text about model visualisation</p>
      </div>
      <div>
        <label htmlFor="forestSelector">Choose Forest: </label>
        <select
          id="dataset-visualisation"
          onChange={(e) => handleForestChange(e.target.value as ForestName)}
        >
          {Object.keys(forestFiles).map((forest) => (
            <option key={forest} value={forest}>
              {forest}
            </option>
          ))}
        </select>
      </div>
      <div style={{ display: "flex", gap: "1rem" }}>
        <div ref={aerialRef} style={{ width: "50%", height: "80vh", border: "1px solid black" }}></div>
        <div ref={groundRef} style={{ width: "50%", height: "80vh", border: "1px solid black" }}></div>
      </div>
    </div>
  );
};

export default DatasetVisualisation;
