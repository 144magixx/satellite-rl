"""Natural compact T0 prefixes and optional appended fields."""
GROUPS={'S00':(False,False),'S10':(True,False),'S01':(False,True),'S11':(True,True)}
def state_width(group):
    width,relations=GROUPS[group]
    return 207+int(width)+200*int(relations)
def observation_schema(group):
    width,relations=GROUPS[group]
    fields=dict(geo=(0,2),rate=(2,3),power=(3,4),interference=(4,104),occupancy=(104,204),global_context=(204,207))
    offset=207
    if width:fields['width']=(offset,offset+1);offset+=1
    if relations:fields.update(out=(offset,offset+100),future_in=(offset+100,offset+200))
    return dict(version='T0WidthCompactV1',group=group,state_width=state_width(group),fields=fields)
