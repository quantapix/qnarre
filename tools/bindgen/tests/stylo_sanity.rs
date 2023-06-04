// Don't want to copy that nasty `cfg` below...
#[allow(unused_extern_crates)]
extern crate bindgen;

/// A sanity test that we can generate bindings for Stylo.
///
/// We don't assert on expected output because its just too big. The output will
/// change too often, and it won't be clear what is going on at a glance, unlike
/// the other tests with smaller input headers.
///
/// This test is relatively slow, so we also only run it in release mode.
///
/// Finally, uncomment the `panic!` at the bottom of the test to get logs timing
/// how long bindings generation takes for Stylo. Stylo bindings generation
/// takes too long to be a proper `#[bench]`.
#[test]
#[cfg(not(any(
    debug_assertions,
    feature = "__testing_only_extra_assertions",
)))]
#[cfg(any(
    feature = "__testing_only_libclang_5",
    feature = "__testing_only_libclang_9"
))]
fn sanity_check_can_generate_stylo_bindings() {
    use std::time::Instant;

    let then = Instant::now();

    bindgen::builder()
        .time_phases(true)
        .disable_header_comment()
        .header(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/stylo.hpp"))
        .allowlist_function("Servo_.*")
        .allowlist_function("Gecko_.*")
        .blocklist_type("nsACString_internal")
        .blocklist_type("nsAString_internal")
        .blocklist_type("mozilla::css::URLValue")
        .blocklist_type("RawGeckoAnimationPropertySegment")
        .blocklist_type("RawGeckoComputedTiming")
        .blocklist_type("RawGeckoDocument")
        .blocklist_type("RawGeckoElement")
        .blocklist_type("RawGeckoKeyframeList")
        .blocklist_type("RawGeckoComputedKeyframeValuesList")
        .blocklist_type("RawGeckoFontFaceRuleList")
        .blocklist_type("RawGeckoNode")
        .blocklist_type("RawGeckoAnimationValueList")
        .blocklist_type("RawServoAnimationValue")
        .blocklist_type("RawServoAnimationValueMap")
        .blocklist_type("RawServoDeclarationBlock")
        .blocklist_type("RawGeckoPresContext")
        .blocklist_type("RawGeckoPresContextOwned")
        .blocklist_type("RawGeckoStyleAnimationList")
        .blocklist_type("RawGeckoURLExtraData")
        .blocklist_type("RefPtr")
        .blocklist_type("CSSPseudoClassType")
        .blocklist_type("TraversalRootBehavior")
        .blocklist_type("ComputedTimingFunction_BeforeFlag")
        .blocklist_type("FontFamilyList")
        .blocklist_type("FontFamilyType")
        .blocklist_type("Keyframe")
        .blocklist_type("ServoBundledURI")
        .blocklist_type("ServoElementSnapshot")
        .blocklist_type("SheetParsingMode")
        .blocklist_type("StyleBasicShape")
        .blocklist_type("StyleBasicShapeType")
        .blocklist_type("StyleShapeSource")
        .blocklist_type("nsCSSFontFaceRule")
        .blocklist_type("nsCSSKeyword")
        .blocklist_type("nsCSSPropertyID")
        .blocklist_type("nsCSSShadowArray")
        .blocklist_type("nsCSSUnit")
        .blocklist_type("nsCSSValue")
        .blocklist_type("nsCSSValueSharedList")
        .blocklist_type("nsChangeHint")
        .blocklist_type("nsCursorImage")
        .blocklist_type("nsFont")
        .blocklist_type("nsIAtom")
        .blocklist_type("nsMediaFeature")
        .blocklist_type("nsRestyleHint")
        .blocklist_type("nsStyleBackground")
        .blocklist_type("nsStyleBorder")
        .blocklist_type("nsStyleColor")
        .blocklist_type("nsStyleColumn")
        .blocklist_type("nsStyleContent")
        .blocklist_type("nsStyleContentData")
        .blocklist_type("nsStyleContentType")
        .blocklist_type("nsStyleContext")
        .blocklist_type("nsStyleCoord")
        .blocklist_type("nsStyleCoord_Calc")
        .blocklist_type("nsStyleCoord_CalcValue")
        .blocklist_type("nsStyleDisplay")
        .blocklist_type("nsStyleEffects")
        .blocklist_type("nsStyleFilter")
        .blocklist_type("nsStyleFont")
        .blocklist_type("nsStyleGradient")
        .blocklist_type("nsStyleGradientStop")
        .blocklist_type("nsStyleImage")
        .blocklist_type("nsStyleImageLayers")
        .blocklist_type("nsStyleImageLayers_Layer")
        .blocklist_type("nsStyleImageLayers_LayerType")
        .blocklist_type("nsStyleImageRequest")
        .blocklist_type("nsStyleList")
        .blocklist_type("nsStyleMargin")
        .blocklist_type("nsStyleOutline")
        .blocklist_type("nsStylePadding")
        .blocklist_type("nsStylePosition")
        .blocklist_type("nsStyleQuoteValues")
        .blocklist_type("nsStyleSVG")
        .blocklist_type("nsStyleSVGPaint")
        .blocklist_type("nsStyleSVGReset")
        .blocklist_type("nsStyleTable")
        .blocklist_type("nsStyleTableBorder")
        .blocklist_type("nsStyleText")
        .blocklist_type("nsStyleTextReset")
        .blocklist_type("nsStyleUIReset")
        .blocklist_type("nsStyleUnion")
        .blocklist_type("nsStyleUnit")
        .blocklist_type("nsStyleUserInterface")
        .blocklist_type("nsStyleVariables")
        .blocklist_type("nsStyleVisibility")
        .blocklist_type("nsStyleXUL")
        .blocklist_type("nsTimingFunction")
        .blocklist_type("nscolor")
        .blocklist_type("nscoord")
        .blocklist_type("nsresult")
        .blocklist_type("Loader")
        .blocklist_type("ServoStyleSheet")
        .blocklist_type("EffectCompositor_CascadeLevel")
        .blocklist_type("UpdateAnimationsTasks")
        .blocklist_type("nsTArrayBorrowed_uintptr_t")
        .blocklist_type("ServoCssRulesStrong")
        .blocklist_type("ServoCssRulesBorrowed")
        .blocklist_type("ServoCssRulesBorrowedOrNull")
        .blocklist_type("ServoCssRules")
        .blocklist_type("RawServoStyleSheetStrong")
        .blocklist_type("RawServoStyleSheetBorrowed")
        .blocklist_type("RawServoStyleSheetBorrowedOrNull")
        .blocklist_type("RawServoStyleSheet")
        .blocklist_type("ServoComputedValuesStrong")
        .blocklist_type("ServoComputedValuesBorrowed")
        .blocklist_type("ServoComputedValuesBorrowedOrNull")
        .blocklist_type("ServoComputedValues")
        .blocklist_type("RawServoDeclarationBlockStrong")
        .blocklist_type("RawServoDeclarationBlockBorrowed")
        .blocklist_type("RawServoDeclarationBlockBorrowedOrNull")
        .blocklist_type("RawServoStyleRuleStrong")
        .blocklist_type("RawServoStyleRuleBorrowed")
        .blocklist_type("RawServoStyleRuleBorrowedOrNull")
        .blocklist_type("RawServoStyleRule")
        .blocklist_type("RawServoImportRuleStrong")
        .blocklist_type("RawServoImportRuleBorrowed")
        .blocklist_type("RawServoImportRuleBorrowedOrNull")
        .blocklist_type("RawServoImportRule")
        .blocklist_type("RawServoAnimationValueStrong")
        .blocklist_type("RawServoAnimationValueBorrowed")
        .blocklist_type("RawServoAnimationValueBorrowedOrNull")
        .blocklist_type("RawServoAnimationValueMapStrong")
        .blocklist_type("RawServoAnimationValueMapBorrowed")
        .blocklist_type("RawServoAnimationValueMapBorrowedOrNull")
        .blocklist_type("RawServoMediaListStrong")
        .blocklist_type("RawServoMediaListBorrowed")
        .blocklist_type("RawServoMediaListBorrowedOrNull")
        .blocklist_type("RawServoMediaList")
        .blocklist_type("RawServoMediaRuleStrong")
        .blocklist_type("RawServoMediaRuleBorrowed")
        .blocklist_type("RawServoMediaRuleBorrowedOrNull")
        .blocklist_type("RawServoMediaRule")
        .blocklist_type("RawServoNamespaceRuleStrong")
        .blocklist_type("RawServoNamespaceRuleBorrowed")
        .blocklist_type("RawServoNamespaceRuleBorrowedOrNull")
        .blocklist_type("RawServoNamespaceRule")
        .blocklist_type("RawServoStyleSetOwned")
        .blocklist_type("RawServoStyleSetOwnedOrNull")
        .blocklist_type("RawServoStyleSetBorrowed")
        .blocklist_type("RawServoStyleSetBorrowedOrNull")
        .blocklist_type("RawServoStyleSetBorrowedMut")
        .blocklist_type("RawServoStyleSetBorrowedMutOrNull")
        .blocklist_type("RawServoStyleSet")
        .blocklist_type("StyleChildrenIteratorOwned")
        .blocklist_type("StyleChildrenIteratorOwnedOrNull")
        .blocklist_type("StyleChildrenIteratorBorrowed")
        .blocklist_type("StyleChildrenIteratorBorrowedOrNull")
        .blocklist_type("StyleChildrenIteratorBorrowedMut")
        .blocklist_type("StyleChildrenIteratorBorrowedMutOrNull")
        .blocklist_type("StyleChildrenIterator")
        .blocklist_type("ServoElementSnapshotOwned")
        .blocklist_type("ServoElementSnapshotOwnedOrNull")
        .blocklist_type("ServoElementSnapshotBorrowed")
        .blocklist_type("ServoElementSnapshotBorrowedOrNull")
        .blocklist_type("ServoElementSnapshotBorrowedMut")
        .blocklist_type("ServoElementSnapshotBorrowedMutOrNull")
        .blocklist_type("RawGeckoNodeBorrowed")
        .blocklist_type("RawGeckoNodeBorrowedOrNull")
        .blocklist_type("RawGeckoElementBorrowed")
        .blocklist_type("RawGeckoElementBorrowedOrNull")
        .blocklist_type("RawGeckoDocumentBorrowed")
        .blocklist_type("RawGeckoDocumentBorrowedOrNull")
        .blocklist_type("RawServoDeclarationBlockStrongBorrowed")
        .blocklist_type("RawServoDeclarationBlockStrongBorrowedOrNull")
        .blocklist_type("RawGeckoPresContextBorrowed")
        .blocklist_type("RawGeckoPresContextBorrowedOrNull")
        .blocklist_type("RawGeckoStyleAnimationListBorrowed")
        .blocklist_type("RawGeckoStyleAnimationListBorrowedOrNull")
        .blocklist_type("nsCSSValueBorrowed")
        .blocklist_type("nsCSSValueBorrowedOrNull")
        .blocklist_type("nsCSSValueBorrowedMut")
        .blocklist_type("nsCSSValueBorrowedMutOrNull")
        .blocklist_type("nsTimingFunctionBorrowed")
        .blocklist_type("nsTimingFunctionBorrowedOrNull")
        .blocklist_type("nsTimingFunctionBorrowedMut")
        .blocklist_type("nsTimingFunctionBorrowedMutOrNull")
        .blocklist_type("RawGeckoAnimationPropertySegmentBorrowed")
        .blocklist_type("RawGeckoAnimationPropertySegmentBorrowedOrNull")
        .blocklist_type("RawGeckoAnimationPropertySegmentBorrowedMut")
        .blocklist_type("RawGeckoAnimationPropertySegmentBorrowedMutOrNull")
        .blocklist_type("RawGeckoAnimationValueListBorrowed")
        .blocklist_type("RawGeckoAnimationValueListBorrowedOrNull")
        .blocklist_type("RawGeckoAnimationValueListBorrowedMut")
        .blocklist_type("RawGeckoAnimationValueListBorrowedMutOrNull")
        .blocklist_type("RawGeckoComputedTimingBorrowed")
        .blocklist_type("RawGeckoComputedTimingBorrowedOrNull")
        .blocklist_type("RawGeckoComputedTimingBorrowedMut")
        .blocklist_type("RawGeckoComputedTimingBorrowedMutOrNull")
        .blocklist_type("RawGeckoKeyframeListBorrowed")
        .blocklist_type("RawGeckoKeyframeListBorrowedOrNull")
        .blocklist_type("RawGeckoKeyframeListBorrowedMut")
        .blocklist_type("RawGeckoKeyframeListBorrowedMutOrNull")
        .blocklist_type("RawGeckoComputedKeyframeValuesListBorrowed")
        .blocklist_type("RawGeckoComputedKeyframeValuesListBorrowedOrNull")
        .blocklist_type("RawGeckoComputedKeyframeValuesListBorrowedMut")
        .blocklist_type("RawGeckoComputedKeyframeValuesListBorrowedMutOrNull")
        .blocklist_type("RawGeckoFontFaceRuleListBorrowed")
        .blocklist_type("RawGeckoFontFaceRuleListBorrowedOrNull")
        .blocklist_type("RawGeckoFontFaceRuleListBorrowedMut")
        .blocklist_type("RawGeckoFontFaceRuleListBorrowedMutOrNull")
        .raw_line(r#"pub use nsstring::{nsACString, nsAString, nsString};"#)
        .raw_line(r#"type nsACString_internal = nsACString;"#)
        .raw_line(r#"type nsAString_internal = nsAString;"#)
        .raw_line(r#"use gecko_bindings::structs::mozilla::css::URLValue;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoAnimationPropertySegment;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoComputedTiming;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoDocument;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoElement;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoKeyframeList;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoComputedKeyframeValuesList;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoFontFaceRuleList;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoNode;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoAnimationValueList;"#)
        .raw_line(r#"use gecko_bindings::structs::RawServoAnimationValue;"#)
        .raw_line(r#"use gecko_bindings::structs::RawServoAnimationValueMap;"#)
        .raw_line(r#"use gecko_bindings::structs::RawServoDeclarationBlock;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoPresContext;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoPresContextOwned;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoStyleAnimationList;"#)
        .raw_line(r#"use gecko_bindings::structs::RawGeckoURLExtraData;"#)
        .raw_line(r#"use gecko_bindings::structs::RefPtr;"#)
        .raw_line(r#"use gecko_bindings::structs::CSSPseudoClassType;"#)
        .raw_line(r#"use gecko_bindings::structs::TraversalRootBehavior;"#)
        .raw_line(r#"use gecko_bindings::structs::ComputedTimingFunction_BeforeFlag;"#)
        .raw_line(r#"use gecko_bindings::structs::FontFamilyList;"#)
        .raw_line(r#"use gecko_bindings::structs::FontFamilyType;"#)
        .raw_line(r#"use gecko_bindings::structs::Keyframe;"#)
        .raw_line(r#"use gecko_bindings::structs::ServoBundledURI;"#)
        .raw_line(r#"use gecko_bindings::structs::ServoElementSnapshot;"#)
        .raw_line(r#"use gecko_bindings::structs::SheetParsingMode;"#)
        .raw_line(r#"use gecko_bindings::structs::StyleBasicShape;"#)
        .raw_line(r#"use gecko_bindings::structs::StyleBasicShapeType;"#)
        .raw_line(r#"use gecko_bindings::structs::StyleShapeSource;"#)
        .raw_line(r#"use gecko_bindings::structs::nsCSSFontFaceRule;"#)
        .raw_line(r#"use gecko_bindings::structs::nsCSSKeyword;"#)
        .raw_line(r#"use gecko_bindings::structs::nsCSSPropertyID;"#)
        .raw_line(r#"use gecko_bindings::structs::nsCSSShadowArray;"#)
        .raw_line(r#"use gecko_bindings::structs::nsCSSUnit;"#)
        .raw_line(r#"use gecko_bindings::structs::nsCSSValue;"#)
        .raw_line(r#"use gecko_bindings::structs::nsCSSValueSharedList;"#)
        .raw_line(r#"use gecko_bindings::structs::nsChangeHint;"#)
        .raw_line(r#"use gecko_bindings::structs::nsCursorImage;"#)
        .raw_line(r#"use gecko_bindings::structs::nsFont;"#)
        .raw_line(r#"use gecko_bindings::structs::nsIAtom;"#)
        .raw_line(r#"use gecko_bindings::structs::nsMediaFeature;"#)
        .raw_line(r#"use gecko_bindings::structs::nsRestyleHint;"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleBackground;"#)
        .raw_line(r#"unsafe impl Send for nsStyleBackground {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleBackground {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleBorder;"#)
        .raw_line(r#"unsafe impl Send for nsStyleBorder {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleBorder {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleColor;"#)
        .raw_line(r#"unsafe impl Send for nsStyleColor {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleColor {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleColumn;"#)
        .raw_line(r#"unsafe impl Send for nsStyleColumn {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleColumn {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleContent;"#)
        .raw_line(r#"unsafe impl Send for nsStyleContent {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleContent {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleContentData;"#)
        .raw_line(r#"unsafe impl Send for nsStyleContentData {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleContentData {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleContentType;"#)
        .raw_line(r#"unsafe impl Send for nsStyleContentType {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleContentType {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleContext;"#)
        .raw_line(r#"unsafe impl Send for nsStyleContext {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleContext {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleCoord;"#)
        .raw_line(r#"unsafe impl Send for nsStyleCoord {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleCoord {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleCoord_Calc;"#)
        .raw_line(r#"unsafe impl Send for nsStyleCoord_Calc {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleCoord_Calc {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleCoord_CalcValue;"#)
        .raw_line(r#"unsafe impl Send for nsStyleCoord_CalcValue {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleCoord_CalcValue {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleDisplay;"#)
        .raw_line(r#"unsafe impl Send for nsStyleDisplay {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleDisplay {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleEffects;"#)
        .raw_line(r#"unsafe impl Send for nsStyleEffects {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleEffects {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleFilter;"#)
        .raw_line(r#"unsafe impl Send for nsStyleFilter {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleFilter {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleFont;"#)
        .raw_line(r#"unsafe impl Send for nsStyleFont {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleFont {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleGradient;"#)
        .raw_line(r#"unsafe impl Send for nsStyleGradient {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleGradient {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleGradientStop;"#)
        .raw_line(r#"unsafe impl Send for nsStyleGradientStop {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleGradientStop {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleImage;"#)
        .raw_line(r#"unsafe impl Send for nsStyleImage {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleImage {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleImageLayers;"#)
        .raw_line(r#"unsafe impl Send for nsStyleImageLayers {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleImageLayers {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleImageLayers_Layer;"#)
        .raw_line(r#"unsafe impl Send for nsStyleImageLayers_Layer {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleImageLayers_Layer {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleImageLayers_LayerType;"#)
        .raw_line(r#"unsafe impl Send for nsStyleImageLayers_LayerType {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleImageLayers_LayerType {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleImageRequest;"#)
        .raw_line(r#"unsafe impl Send for nsStyleImageRequest {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleImageRequest {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleList;"#)
        .raw_line(r#"unsafe impl Send for nsStyleList {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleList {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleMargin;"#)
        .raw_line(r#"unsafe impl Send for nsStyleMargin {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleMargin {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleOutline;"#)
        .raw_line(r#"unsafe impl Send for nsStyleOutline {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleOutline {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStylePadding;"#)
        .raw_line(r#"unsafe impl Send for nsStylePadding {}"#)
        .raw_line(r#"unsafe impl Sync for nsStylePadding {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStylePosition;"#)
        .raw_line(r#"unsafe impl Send for nsStylePosition {}"#)
        .raw_line(r#"unsafe impl Sync for nsStylePosition {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleQuoteValues;"#)
        .raw_line(r#"unsafe impl Send for nsStyleQuoteValues {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleQuoteValues {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleSVG;"#)
        .raw_line(r#"unsafe impl Send for nsStyleSVG {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleSVG {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleSVGPaint;"#)
        .raw_line(r#"unsafe impl Send for nsStyleSVGPaint {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleSVGPaint {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleSVGReset;"#)
        .raw_line(r#"unsafe impl Send for nsStyleSVGReset {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleSVGReset {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleTable;"#)
        .raw_line(r#"unsafe impl Send for nsStyleTable {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleTable {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleTableBorder;"#)
        .raw_line(r#"unsafe impl Send for nsStyleTableBorder {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleTableBorder {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleText;"#)
        .raw_line(r#"unsafe impl Send for nsStyleText {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleText {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleTextReset;"#)
        .raw_line(r#"unsafe impl Send for nsStyleTextReset {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleTextReset {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleUIReset;"#)
        .raw_line(r#"unsafe impl Send for nsStyleUIReset {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleUIReset {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleUnion;"#)
        .raw_line(r#"unsafe impl Send for nsStyleUnion {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleUnion {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleUnit;"#)
        .raw_line(r#"unsafe impl Send for nsStyleUnit {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleUnit {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleUserInterface;"#)
        .raw_line(r#"unsafe impl Send for nsStyleUserInterface {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleUserInterface {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleVariables;"#)
        .raw_line(r#"unsafe impl Send for nsStyleVariables {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleVariables {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleVisibility;"#)
        .raw_line(r#"unsafe impl Send for nsStyleVisibility {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleVisibility {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsStyleXUL;"#)
        .raw_line(r#"unsafe impl Send for nsStyleXUL {}"#)
        .raw_line(r#"unsafe impl Sync for nsStyleXUL {}"#)
        .raw_line(r#"use gecko_bindings::structs::nsTimingFunction;"#)
        .raw_line(r#"use gecko_bindings::structs::nscolor;"#)
        .raw_line(r#"use gecko_bindings::structs::nscoord;"#)
        .raw_line(r#"use gecko_bindings::structs::nsresult;"#)
        .raw_line(r#"use gecko_bindings::structs::Loader;"#)
        .raw_line(r#"use gecko_bindings::structs::ServoStyleSheet;"#)
        .raw_line(r#"use gecko_bindings::structs::EffectCompositor_CascadeLevel;"#)
        .raw_line(r#"use gecko_bindings::structs::UpdateAnimationsTasks;"#)
        .raw_line(r#"pub type nsTArrayBorrowed_uintptr_t<'a> = &'a mut ::gecko_bindings::structs::nsTArray<usize>;"#)
        .raw_line(r#"pub type ServoCssRulesStrong = ::gecko_bindings::sugar::ownership::Strong<ServoCssRules>;"#)
        .raw_line(r#"pub type ServoCssRulesBorrowed<'a> = &'a ServoCssRules;"#)
        .raw_line(r#"pub type ServoCssRulesBorrowedOrNull<'a> = Option<&'a ServoCssRules>;"#)
        .raw_line(r#"enum ServoCssRulesVoid { }"#)
        .raw_line(r#"pub struct ServoCssRules(ServoCssRulesVoid);"#)
        .raw_line(r#"pub type RawServoStyleSheetStrong = ::gecko_bindings::sugar::ownership::Strong<RawServoStyleSheet>;"#)
        .raw_line(r#"pub type RawServoStyleSheetBorrowed<'a> = &'a RawServoStyleSheet;"#)
        .raw_line(r#"pub type RawServoStyleSheetBorrowedOrNull<'a> = Option<&'a RawServoStyleSheet>;"#)
        .raw_line(r#"enum RawServoStyleSheetVoid { }"#)
        .raw_line(r#"pub struct RawServoStyleSheet(RawServoStyleSheetVoid);"#)
        .raw_line(r#"pub type ServoComputedValuesStrong = ::gecko_bindings::sugar::ownership::Strong<ServoComputedValues>;"#)
        .raw_line(r#"pub type ServoComputedValuesBorrowed<'a> = &'a ServoComputedValues;"#)
        .raw_line(r#"pub type ServoComputedValuesBorrowedOrNull<'a> = Option<&'a ServoComputedValues>;"#)
        .raw_line(r#"enum ServoComputedValuesVoid { }"#)
        .raw_line(r#"pub struct ServoComputedValues(ServoComputedValuesVoid);"#)
        .raw_line(r#"pub type RawServoDeclarationBlockStrong = ::gecko_bindings::sugar::ownership::Strong<RawServoDeclarationBlock>;"#)
        .raw_line(r#"pub type RawServoDeclarationBlockBorrowed<'a> = &'a RawServoDeclarationBlock;"#)
        .raw_line(r#"pub type RawServoDeclarationBlockBorrowedOrNull<'a> = Option<&'a RawServoDeclarationBlock>;"#)
        .raw_line(r#"pub type RawServoStyleRuleStrong = ::gecko_bindings::sugar::ownership::Strong<RawServoStyleRule>;"#)
        .raw_line(r#"pub type RawServoStyleRuleBorrowed<'a> = &'a RawServoStyleRule;"#)
        .raw_line(r#"pub type RawServoStyleRuleBorrowedOrNull<'a> = Option<&'a RawServoStyleRule>;"#)
        .raw_line(r#"enum RawServoStyleRuleVoid { }"#)
        .raw_line(r#"pub struct RawServoStyleRule(RawServoStyleRuleVoid);"#)
        .raw_line(r#"pub type RawServoImportRuleStrong = ::gecko_bindings::sugar::ownership::Strong<RawServoImportRule>;"#)
        .raw_line(r#"pub type RawServoImportRuleBorrowed<'a> = &'a RawServoImportRule;"#)
        .raw_line(r#"pub type RawServoImportRuleBorrowedOrNull<'a> = Option<&'a RawServoImportRule>;"#)
        .raw_line(r#"enum RawServoImportRuleVoid { }"#)
        .raw_line(r#"pub struct RawServoImportRule(RawServoImportRuleVoid);"#)
        .raw_line(r#"pub type RawServoAnimationValueStrong = ::gecko_bindings::sugar::ownership::Strong<RawServoAnimationValue>;"#)
        .raw_line(r#"pub type RawServoAnimationValueBorrowed<'a> = &'a RawServoAnimationValue;"#)
        .raw_line(r#"pub type RawServoAnimationValueBorrowedOrNull<'a> = Option<&'a RawServoAnimationValue>;"#)
        .raw_line(r#"pub type RawServoAnimationValueMapStrong = ::gecko_bindings::sugar::ownership::Strong<RawServoAnimationValueMap>;"#)
        .raw_line(r#"pub type RawServoAnimationValueMapBorrowed<'a> = &'a RawServoAnimationValueMap;"#)
        .raw_line(r#"pub type RawServoAnimationValueMapBorrowedOrNull<'a> = Option<&'a RawServoAnimationValueMap>;"#)
        .raw_line(r#"pub type RawServoMediaListStrong = ::gecko_bindings::sugar::ownership::Strong<RawServoMediaList>;"#)
        .raw_line(r#"pub type RawServoMediaListBorrowed<'a> = &'a RawServoMediaList;"#)
        .raw_line(r#"pub type RawServoMediaListBorrowedOrNull<'a> = Option<&'a RawServoMediaList>;"#)
        .raw_line(r#"enum RawServoMediaListVoid { }"#)
        .raw_line(r#"pub struct RawServoMediaList(RawServoMediaListVoid);"#)
        .raw_line(r#"pub type RawServoMediaRuleStrong = ::gecko_bindings::sugar::ownership::Strong<RawServoMediaRule>;"#)
        .raw_line(r#"pub type RawServoMediaRuleBorrowed<'a> = &'a RawServoMediaRule;"#)
        .raw_line(r#"pub type RawServoMediaRuleBorrowedOrNull<'a> = Option<&'a RawServoMediaRule>;"#)
        .raw_line(r#"enum RawServoMediaRuleVoid { }"#)
        .raw_line(r#"pub struct RawServoMediaRule(RawServoMediaRuleVoid);"#)
        .raw_line(r#"pub type RawServoNamespaceRuleStrong = ::gecko_bindings::sugar::ownership::Strong<RawServoNamespaceRule>;"#)
        .raw_line(r#"pub type RawServoNamespaceRuleBorrowed<'a> = &'a RawServoNamespaceRule;"#)
        .raw_line(r#"pub type RawServoNamespaceRuleBorrowedOrNull<'a> = Option<&'a RawServoNamespaceRule>;"#)
        .raw_line(r#"enum RawServoNamespaceRuleVoid { }"#)
        .raw_line(r#"pub struct RawServoNamespaceRule(RawServoNamespaceRuleVoid);"#)
        .raw_line(r#"pub type RawServoStyleSetOwned = ::gecko_bindings::sugar::ownership::Owned<RawServoStyleSet>;"#)
        .raw_line(r#"pub type RawServoStyleSetOwnedOrNull = ::gecko_bindings::sugar::ownership::OwnedOrNull<RawServoStyleSet>;"#)
        .raw_line(r#"pub type RawServoStyleSetBorrowed<'a> = &'a RawServoStyleSet;"#)
        .raw_line(r#"pub type RawServoStyleSetBorrowedOrNull<'a> = Option<&'a RawServoStyleSet>;"#)
        .raw_line(r#"pub type RawServoStyleSetBorrowedMut<'a> = &'a mut RawServoStyleSet;"#)
        .raw_line(r#"pub type RawServoStyleSetBorrowedMutOrNull<'a> = Option<&'a mut RawServoStyleSet>;"#)
        .raw_line(r#"enum RawServoStyleSetVoid { }"#)
        .raw_line(r#"pub struct RawServoStyleSet(RawServoStyleSetVoid);"#)
        .raw_line(r#"pub type StyleChildrenIteratorOwned = ::gecko_bindings::sugar::ownership::Owned<StyleChildrenIterator>;"#)
        .raw_line(r#"pub type StyleChildrenIteratorOwnedOrNull = ::gecko_bindings::sugar::ownership::OwnedOrNull<StyleChildrenIterator>;"#)
        .raw_line(r#"pub type StyleChildrenIteratorBorrowed<'a> = &'a StyleChildrenIterator;"#)
        .raw_line(r#"pub type StyleChildrenIteratorBorrowedOrNull<'a> = Option<&'a StyleChildrenIterator>;"#)
        .raw_line(r#"pub type StyleChildrenIteratorBorrowedMut<'a> = &'a mut StyleChildrenIterator;"#)
        .raw_line(r#"pub type StyleChildrenIteratorBorrowedMutOrNull<'a> = Option<&'a mut StyleChildrenIterator>;"#)
        .raw_line(r#"enum StyleChildrenIteratorVoid { }"#)
        .raw_line(r#"pub struct StyleChildrenIterator(StyleChildrenIteratorVoid);"#)
        .raw_line(r#"pub type ServoElementSnapshotOwned = ::gecko_bindings::sugar::ownership::Owned<ServoElementSnapshot>;"#)
        .raw_line(r#"pub type ServoElementSnapshotOwnedOrNull = ::gecko_bindings::sugar::ownership::OwnedOrNull<ServoElementSnapshot>;"#)
        .raw_line(r#"pub type ServoElementSnapshotBorrowed<'a> = &'a ServoElementSnapshot;"#)
        .raw_line(r#"pub type ServoElementSnapshotBorrowedOrNull<'a> = Option<&'a ServoElementSnapshot>;"#)
        .raw_line(r#"pub type ServoElementSnapshotBorrowedMut<'a> = &'a mut ServoElementSnapshot;"#)
        .raw_line(r#"pub type ServoElementSnapshotBorrowedMutOrNull<'a> = Option<&'a mut ServoElementSnapshot>;"#)
        .raw_line(r#"pub type RawGeckoNodeBorrowed<'a> = &'a RawGeckoNode;"#)
        .raw_line(r#"pub type RawGeckoNodeBorrowedOrNull<'a> = Option<&'a RawGeckoNode>;"#)
        .raw_line(r#"pub type RawGeckoElementBorrowed<'a> = &'a RawGeckoElement;"#)
        .raw_line(r#"pub type RawGeckoElementBorrowedOrNull<'a> = Option<&'a RawGeckoElement>;"#)
        .raw_line(r#"pub type RawGeckoDocumentBorrowed<'a> = &'a RawGeckoDocument;"#)
        .raw_line(r#"pub type RawGeckoDocumentBorrowedOrNull<'a> = Option<&'a RawGeckoDocument>;"#)
        .raw_line(r#"pub type RawServoDeclarationBlockStrongBorrowed<'a> = &'a RawServoDeclarationBlockStrong;"#)
        .raw_line(r#"pub type RawServoDeclarationBlockStrongBorrowedOrNull<'a> = Option<&'a RawServoDeclarationBlockStrong>;"#)
        .raw_line(r#"pub type RawGeckoPresContextBorrowed<'a> = &'a RawGeckoPresContext;"#)
        .raw_line(r#"pub type RawGeckoPresContextBorrowedOrNull<'a> = Option<&'a RawGeckoPresContext>;"#)
        .raw_line(r#"pub type RawGeckoStyleAnimationListBorrowed<'a> = &'a RawGeckoStyleAnimationList;"#)
        .raw_line(r#"pub type RawGeckoStyleAnimationListBorrowedOrNull<'a> = Option<&'a RawGeckoStyleAnimationList>;"#)
        .raw_line(r#"pub type nsCSSValueBorrowed<'a> = &'a nsCSSValue;"#)
        .raw_line(r#"pub type nsCSSValueBorrowedOrNull<'a> = Option<&'a nsCSSValue>;"#)
        .raw_line(r#"pub type nsCSSValueBorrowedMut<'a> = &'a mut nsCSSValue;"#)
        .raw_line(r#"pub type nsCSSValueBorrowedMutOrNull<'a> = Option<&'a mut nsCSSValue>;"#)
        .raw_line(r#"pub type nsTimingFunctionBorrowed<'a> = &'a nsTimingFunction;"#)
        .raw_line(r#"pub type nsTimingFunctionBorrowedOrNull<'a> = Option<&'a nsTimingFunction>;"#)
        .raw_line(r#"pub type nsTimingFunctionBorrowedMut<'a> = &'a mut nsTimingFunction;"#)
        .raw_line(r#"pub type nsTimingFunctionBorrowedMutOrNull<'a> = Option<&'a mut nsTimingFunction>;"#)
        .raw_line(r#"pub type RawGeckoAnimationPropertySegmentBorrowed<'a> = &'a RawGeckoAnimationPropertySegment;"#)
        .raw_line(r#"pub type RawGeckoAnimationPropertySegmentBorrowedOrNull<'a> = Option<&'a RawGeckoAnimationPropertySegment>;"#)
        .raw_line(r#"pub type RawGeckoAnimationPropertySegmentBorrowedMut<'a> = &'a mut RawGeckoAnimationPropertySegment;"#)
        .raw_line(r#"pub type RawGeckoAnimationPropertySegmentBorrowedMutOrNull<'a> = Option<&'a mut RawGeckoAnimationPropertySegment>;"#)
        .raw_line(r#"pub type RawGeckoAnimationValueListBorrowed<'a> = &'a RawGeckoAnimationValueList;"#)
        .raw_line(r#"pub type RawGeckoAnimationValueListBorrowedOrNull<'a> = Option<&'a RawGeckoAnimationValueList>;"#)
        .raw_line(r#"pub type RawGeckoAnimationValueListBorrowedMut<'a> = &'a mut RawGeckoAnimationValueList;"#)
        .raw_line(r#"pub type RawGeckoAnimationValueListBorrowedMutOrNull<'a> = Option<&'a mut RawGeckoAnimationValueList>;"#)
        .raw_line(r#"pub type RawGeckoComputedTimingBorrowed<'a> = &'a RawGeckoComputedTiming;"#)
        .raw_line(r#"pub type RawGeckoComputedTimingBorrowedOrNull<'a> = Option<&'a RawGeckoComputedTiming>;"#)
        .raw_line(r#"pub type RawGeckoComputedTimingBorrowedMut<'a> = &'a mut RawGeckoComputedTiming;"#)
        .raw_line(r#"pub type RawGeckoComputedTimingBorrowedMutOrNull<'a> = Option<&'a mut RawGeckoComputedTiming>;"#)
        .raw_line(r#"pub type RawGeckoKeyframeListBorrowed<'a> = &'a RawGeckoKeyframeList;"#)
        .raw_line(r#"pub type RawGeckoKeyframeListBorrowedOrNull<'a> = Option<&'a RawGeckoKeyframeList>;"#)
        .raw_line(r#"pub type RawGeckoKeyframeListBorrowedMut<'a> = &'a mut RawGeckoKeyframeList;"#)
        .raw_line(r#"pub type RawGeckoKeyframeListBorrowedMutOrNull<'a> = Option<&'a mut RawGeckoKeyframeList>;"#)
        .raw_line(r#"pub type RawGeckoComputedKeyframeValuesListBorrowed<'a> = &'a RawGeckoComputedKeyframeValuesList;"#)
        .raw_line(r#"pub type RawGeckoComputedKeyframeValuesListBorrowedOrNull<'a> = Option<&'a RawGeckoComputedKeyframeValuesList>;"#)
        .raw_line(r#"pub type RawGeckoComputedKeyframeValuesListBorrowedMut<'a> = &'a mut RawGeckoComputedKeyframeValuesList;"#)
        .raw_line(r#"pub type RawGeckoComputedKeyframeValuesListBorrowedMutOrNull<'a> = Option<&'a mut RawGeckoComputedKeyframeValuesList>;"#)
        .raw_line(r#"pub type RawGeckoFontFaceRuleListBorrowed<'a> = &'a RawGeckoFontFaceRuleList;"#)
        .raw_line(r#"pub type RawGeckoFontFaceRuleListBorrowedOrNull<'a> = Option<&'a RawGeckoFontFaceRuleList>;"#)
        .raw_line(r#"pub type RawGeckoFontFaceRuleListBorrowedMut<'a> = &'a mut RawGeckoFontFaceRuleList;"#)
        .raw_line(r#"pub type RawGeckoFontFaceRuleListBorrowedMutOrNull<'a> = Option<&'a mut RawGeckoFontFaceRuleList>;"#)
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++14")
        .clang_arg("-DTRACING=1")
        .clang_arg("-DIMPL_LIBXUL")
        .clang_arg("-DMOZ_STYLO_BINDINGS=1")
        .clang_arg("-DMOZILLA_INTERNAL_API")
        .clang_arg("-DRUST_BINDGEN")
        .clang_arg("-DMOZ_STYLO")
        .clang_arg("-DOS_POSIX=1")
        .clang_arg("-DOS_LINUX=1")
        .generate()
        .expect("Should generate stylo bindings");

    let now = Instant::now();

    println!();
    println!();
    println!(
        "Generated Stylo bindings in: {:?}",
        now.duration_since(then)
    );
    println!();
    println!();

    // panic!("Uncomment this line to get timing logs");
}
