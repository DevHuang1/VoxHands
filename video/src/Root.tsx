import React from "react";
import {Composition} from "remotion";
import {VOXHANDS_DURATION, VoxHands} from "./VoxHands";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="VoxHands"
        component={VoxHands}
        durationInFrames={VOXHANDS_DURATION}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
