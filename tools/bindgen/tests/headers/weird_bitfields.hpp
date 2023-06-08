// bindgen-flags: --rustified-enum ".*"

// You can guess where this is taken from...
enum nsStyleSVGOpacitySource {
  eStyleSVGOpacitySource_Normal,
  eStyleSVGOpacitySource_ContextFillOpacity,
  eStyleSVGOpacitySource_ContextStrokeOpacity
};

class Weird {
  unsigned int mStrokeDasharrayLength;
  unsigned int bitTest : 16;
  unsigned int bitTest2 : 15;
  unsigned char mClipRule;                  // [inherited]
  unsigned char mColorInterpolation;        // [inherited] see nsStyleConsts.h
  unsigned char mColorInterpolationFilters; // [inherited] see nsStyleConsts.h
  unsigned char mFillRule;                  // [inherited] see nsStyleConsts.h
  unsigned char mImageRendering;            // [inherited] see nsStyleConsts.h
  unsigned char mPaintOrder;                // [inherited] see nsStyleConsts.h
  unsigned char mShapeRendering;            // [inherited] see nsStyleConsts.h
  unsigned char mStrokeLinecap;             // [inherited] see nsStyleConsts.h
  unsigned char mStrokeLinejoin;            // [inherited] see nsStyleConsts.h
  unsigned char mTextAnchor;                // [inherited] see nsStyleConsts.h
  unsigned char mTextRendering;             // [inherited] see nsStyleConsts.h

  nsStyleSVGOpacitySource mFillOpacitySource : 3;
  nsStyleSVGOpacitySource mStrokeOpacitySource : 3;

  bool mStrokeDasharrayFromObject : 1;
  bool mStrokeDashoffsetFromObject : 1;
  bool mStrokeWidthFromObject : 1;
};
