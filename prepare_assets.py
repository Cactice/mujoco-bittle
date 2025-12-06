import xml.etree.ElementTree as ET
import os
import mujoco
import ctypes


def prepare_urdf(
    input_path, output_urdf_path, output_xml_path, base_link_name, joint_names
):
    tree = ET.parse(input_path)
    root = tree.getroot()

    # 1. Add world link and floating joint
    # Check if world link already exists
    if not any(link.get("name") == "world" for link in root.findall("link")):
        world_link = ET.SubElement(root, "link")
        world_link.set("name", "world")

        base_joint = ET.SubElement(root, "joint")
        base_joint.set("name", "base_joint")
        base_joint.set("type", "floating")

        parent = ET.SubElement(base_joint, "parent")
        parent.set("link", "world")

        child = ET.SubElement(base_joint, "child")
        child.set("link", base_link_name)

    # 2. Fix Inertias
    for link in root.findall("link"):
        if link.get("name") == "world":
            continue

        inertial = link.find("inertial")
        if inertial is not None:
            inertia = inertial.find("inertia")
            if inertia is not None:
                ixx = float(inertia.get("ixx", 0))
                if ixx == 0:
                    # Replace with default valid inertia
                    inertia.set("ixx", "0.001")
                    inertia.set("ixy", "0")
                    inertia.set("ixz", "0")
                    inertia.set("iyy", "0.001")
                    inertia.set("iyz", "0")
                    inertia.set("izz", "0.001")

    tree.write(output_urdf_path)
    print(f"Patched URDF saved to {output_urdf_path}")

    # 3. Convert to MJCF using MuJoCo
    try:
        model = mujoco.MjModel.from_xml_path(output_urdf_path)
        mujoco.mj_saveLastXML(output_xml_path, model)
        print(f"Converted MJCF saved to {output_xml_path}")
    except Exception as e:
        print(f"Failed to convert URDF to MJCF: {e}")
        return

    # 4. Inject Actuators and Environment (Floor, Light) into MJCF
    tree_xml = ET.parse(output_xml_path)
    root_xml = tree_xml.getroot()

    # Add visual settings for lighter background
    visual = ET.SubElement(root_xml, "visual")
    headlight = ET.SubElement(visual, "headlight")
    headlight.set("diffuse", "0.6 0.6 0.6")
    headlight.set("ambient", "0.3 0.3 0.3")
    headlight.set("specular", "0 0 0")
    rgba = ET.SubElement(visual, "rgba")
    rgba.set("haze", "0.15 0.25 0.35 1")  # Light blueish haze

    # Add assets (grid texture)
    asset = root_xml.find("asset")
    if asset is None:
        asset = ET.SubElement(root_xml, "asset")

    texture = ET.SubElement(asset, "texture")
    texture.set("type", "2d")
    texture.set("name", "grid")
    texture.set("builtin", "checker")
    texture.set("rgb1", ".1 .2 .3")
    texture.set("rgb2", ".2 .3 .4")
    texture.set("width", "300")
    texture.set("height", "300")
    texture.set("mark", "edge")
    texture.set("markrgb", ".2 .3 .4")

    # Add Skybox
    sky_texture = ET.SubElement(asset, "texture")
    sky_texture.set("type", "skybox")
    sky_texture.set("builtin", "gradient")
    sky_texture.set("rgb1", ".3 .5 .7")
    sky_texture.set("rgb2", "0 0 0")
    sky_texture.set("width", "512")
    sky_texture.set("height", "512")

    material = ET.SubElement(asset, "material")
    material.set("name", "grid")
    material.set("texture", "grid")
    material.set("texrepeat", "1 1")
    material.set("texuniform", "true")
    material.set("reflectance", ".2")

    # Modify "black" material to be lighter (Dark Grey)
    for mat in asset.findall("material"):
        if mat.get("name") == "black":
            mat.set("rgba", "0.3 0.3 0.3 1")

    # Add floor and light to worldbody
    worldbody = root_xml.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root_xml, "worldbody")

    light = ET.SubElement(worldbody, "light")
    light.set("diffuse", ".5 .5 .5")
    light.set("pos", "0 0 3")
    light.set("dir", "0 0 -1")

    floor = ET.SubElement(worldbody, "geom")
    floor.set("name", "floor")
    floor.set("type", "plane")
    floor.set("size", "0 0 .05")
    floor.set("material", "grid")

    actuator = ET.SubElement(root_xml, "actuator")

    # Define actuators for joints
    for joint in joint_names:
        # Check if joint exists in MJCF (it should)
        # Add position servo
        motor = ET.SubElement(actuator, "position")
        motor.set("name", f"{joint}_servo")
        motor.set("joint", joint)
        motor.set("kp", "5")
        motor.set("kv", "0.1")

    tree_xml.write(output_xml_path)
    print(f"Final MJCF with actuators saved to {output_xml_path}")


if __name__ == "__main__":
    # Bittle
    bittle_joints = [
        "shoulder_left",
        "shoulder_right",
        "hip_right",
        "hip_left",
        "elbow_left",
        "elbow_right",
        "knee_right",
        "knee_left",
    ]
    prepare_urdf(
        "assets/bittle_esp32.urdf",
        "assets/bittle_sim.urdf",
        "assets/bittle_final.xml",
        "torso",
        bittle_joints,
    )

    # Humanoid
    humanoid_joints = [
        "waist_joint",
        "torso_to_head_joint",
        "torso_to_right_shoulder_joint",
        "right_elbow_joint",
        "right_wrist_joint",
        "torso_to_left_shoulder_joint",
        "left_elbow_joint",
        "left_wrist_joint",
        "torso_to_right_hip_joint",
        "right_knee_joint",
        "right_ankle_joint",
        "torso_to_left_hip_joint",
        "left_knee_joint",
        "left_ankle_joint",
    ]
    prepare_urdf(
        "assets/humanoid.urdf",
        "assets/humanoid_sim.urdf",
        "assets/humanoid_final.xml",
        "base_link",
        humanoid_joints,
    )
