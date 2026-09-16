import React from "react";
import {Composition} from "remotion";
import {VOXHANDS_DURATION, VoxHands} from "./VoxHands";
import {DECK_SLIDES, Deck} from "./Deck";

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
      <Composition
        id="VoxHandsDeck"
        component={Deck}
        durationInFrames={DECK_SLIDES}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
