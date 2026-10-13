"""Pure union/provenance checks: no geometry or algorithm evaluation."""
def check_record(value,kind,identity,source_sha256):
 allowed={'prepared':('available','unavailable'),'teacher':('available','unavailable'),'proposal':('ready','unavailable'),'decoder':('success','fallback','unavailable','unpublished')}[kind]
 if not isinstance(value,dict)or value.get('status')not in allowed:raise ValueError('invalid '+kind+' discriminator')
 if kind=='prepared':provenance=value.get('identity',{})
 elif value['status']=='unavailable':provenance=value.get('provenance',value)
 else:provenance=value
 if not isinstance(provenance,dict)or any(provenance.get(k)!=v for k,v in identity.items()):raise ValueError(kind+' identity/SHA/role mismatch')
 actual=value.get('source_sha256',provenance.get('source_sha256'))
 if actual!=source_sha256:raise ValueError(kind+' source SHA mismatch')
 if value['status']=='unavailable'and kind!='prepared'and any(k in value for k in ('layout','metrics','slot_ids','scores_nxK_float32_finite','anchor_aircraft_ids','mask','mask_nxK_bool','score_mask_nxK_bool','direction','direction_offsets_Kx3_float64_finite','direction_offset_Kx3','assignment','centers','widths_deg')):raise ValueError('unavailable has ready/layout arrays')
 return value
