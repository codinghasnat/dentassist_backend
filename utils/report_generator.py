import os
import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from io import BytesIO
import base64
from .image_processing import get_disease_color

# Define dental health scoring system
DISEASE_SEVERITY = {
    "Healthy": 0,
    "Caries": 2,
    "Deeper Caries": 4,
    "Periapical Lesion": 3,
    "Impacted": 2,
    "Fractured": 3,
    "BDC/BDR": 2,
    "Unknown": 1
}

DISEASE_RECOMMENDATIONS = {
    "Healthy": "No treatment needed. Continue with regular brushing, flossing, and biannual dental checkups.",
    "Caries": "Early cavity detected. Treatment options include fluoride treatments or small fillings. Schedule a follow-up appointment soon.",
    "Deeper Caries": "Advanced decay detected. Will likely require root canal therapy or extensive restoration. Prompt treatment recommended.",
    "Periapical Lesion": "Infection detected at tooth root. Root canal treatment needed. Contact your dentist promptly.",
    "Impacted": "Impacted tooth detected. Consult with an oral surgeon about extraction or monitoring.",
    "Fractured": "Fractured tooth detected. Treatment depends on severity - may require crown, bonding, or extraction.",
    "BDC/BDR": "Bone defect or resorption detected. Additional imaging and specialist consultation recommended.",
    "Unknown": "Undefined condition detected. Further examination required."
}

def calculate_oral_health_score(teeth_by_disease):
    """
    Calculate an overall oral health score based on teeth conditions.
    Returns a score from 0-100 (higher is better) and a rating category.
    """
    # Ensure teeth_by_disease is valid
    if not teeth_by_disease:
        print("[WARNING] Empty teeth_by_disease in calculate_oral_health_score")
        return 0, "No teeth detected"
    
    # Ensure the structure is as expected
    if not isinstance(teeth_by_disease, dict):
        print(f"[WARNING] teeth_by_disease is not a dictionary: {type(teeth_by_disease)}")
        return 0, "Invalid data format"
    
    total_teeth = sum(len(teeth) for teeth in teeth_by_disease.values())
    if total_teeth == 0:
        return 0, "No teeth detected"
    
    # Calculate weighted severity score
    total_severity = 0
    for disease, teeth in teeth_by_disease.items():
        total_severity += DISEASE_SEVERITY.get(disease, 1) * len(teeth)
    
    # Calculate score (100 = all healthy, 0 = all severe)
    max_possible_severity = 4 * total_teeth  # Max severity score if all teeth had the worst condition
    if max_possible_severity == 0:
        score = 0
    else:
        score = 100 - (total_severity / max_possible_severity) * 100
    
    # Determine rating category
    if score >= 90:
        rating = "Excellent"
    elif score >= 75:
        rating = "Good"
    elif score >= 60:
        rating = "Fair"
    elif score >= 40:
        rating = "Poor"
    else:
        rating = "Critical"
    
    return round(score), rating

def generate_recommendations(teeth_by_disease):
    """Generate personalized recommendations based on detected conditions."""
    recommendations = []
    
    # Validate input
    if not teeth_by_disease or not isinstance(teeth_by_disease, dict):
        print("[WARNING] Invalid teeth_by_disease in generate_recommendations")
        return ["Schedule a comprehensive dental exam to get a proper assessment."]
    
    # Add general recommendation
    recommendations.append("Schedule a follow-up with your dentist to discuss these findings.")
    
    # Add disease-specific recommendations
    for disease in teeth_by_disease.keys():
        if disease in DISEASE_RECOMMENDATIONS:
            recommendations.append(f"{disease}: {DISEASE_RECOMMENDATIONS[disease]}")
    
    # General oral health recommendations
    recommendations.append("Maintain good oral hygiene: Brush twice daily, floss daily, and use mouthwash.")
    recommendations.append("Reduce sugar intake to prevent further decay.")
    
    return recommendations

def decode_base64_image(base64_string):
    """Convert a base64 string to a PIL Image."""
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        
        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data))
    except Exception as e:
        print(f"[ERROR] Failed to decode base64 image: {str(e)}")
        # Return a blank image as fallback
        return Image.new('RGB', (100, 100), color=(200, 200, 200))

def pil_to_reportlab_image(pil_img, width=6*inch, max_height=7*inch):
    """
    Convert PIL image to ReportLab Image object with size constraints.
    Ensures the image will fit on the page by resizing if necessary.
    """
    img_width, img_height = pil_img.size
    aspect_ratio = img_height / img_width

    # Calculate new dimensions that will fit on the page
    new_width = min(width, img_width)
    new_height = new_width * aspect_ratio
    if new_height > max_height:
        new_height = max_height
        new_width = new_height / aspect_ratio

    # Resize the image if needed
    if img_width > new_width or img_height > new_height:
        pil_img = pil_img.resize((int(new_width), int(new_height)), Image.LANCZOS)

    img_byte_arr = BytesIO()
    pil_img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return ReportLabImage(img_byte_arr, width=new_width)

def generate_pdf_report(report_data, output_path):
    """
    Generate a comprehensive dental analysis PDF report.
    
    Args:
        report_data: Dict containing original_image, annotated_image, teeth_by_disease
        output_path: Path where to save the PDF
    """
    # Debug the structure of the incoming data
    print("[DEBUG] Report data keys:", report_data.keys())
    
    # Check if the structure matches what we expect
    if 'original_image' not in report_data or 'annotated_image' not in report_data or 'teeth_by_disease' not in report_data:
        print("[ERROR] Invalid data structure. Converting to expected format.")
        # Try to convert from frontend format if needed
        converted_data = {}
        
        # Handle originalImage vs original_image naming differences
        if 'originalImage' in report_data:
            converted_data['original_image'] = report_data['originalImage']
        elif 'original_image' in report_data:
            converted_data['original_image'] = report_data['original_image']
        else:
            raise ValueError("Missing original image data")
            
        # Handle annotatedImage vs annotated_image naming differences
        if 'annotatedImage' in report_data:
            converted_data['annotated_image'] = report_data['annotatedImage']
        elif 'annotated_image' in report_data:
            converted_data['annotated_image'] = report_data['annotated_image']
        else:
            raise ValueError("Missing annotated image data")
            
        # Handle teethByDisease vs teeth_by_disease naming differences
        if 'teethByDisease' in report_data:
            converted_data['teeth_by_disease'] = report_data['teethByDisease']
        elif 'teeth_by_disease' in report_data:
            converted_data['teeth_by_disease'] = report_data['teeth_by_disease']
        else:
            raise ValueError("Missing teeth disease data")
            
        # Use the converted data
        report_data = converted_data
        
    print("[DEBUG] Using report data with keys:", report_data.keys())
    
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=16
    )
    
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=14,
        spaceAfter=10
    )
    
    subheading_style = ParagraphStyle(
        'Subheading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8
    )
    
    normal_style = styles['Normal']
    
    # Calculate oral health score
    try:
        score, rating = calculate_oral_health_score(report_data['teeth_by_disease'])
        recommendations = generate_recommendations(report_data['teeth_by_disease'])
    except Exception as e:
        print(f"[ERROR] Error calculating health score: {str(e)}")
        score, rating = 0, "Error calculating score"
        recommendations = ["Please consult with your dentist for a professional assessment."]
    
    # Start building the document
    elements = []
    
    # Title
    elements.append(Paragraph("Dental X-Ray Analysis Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y')}", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Summary section
    elements.append(Paragraph("Summary", heading_style))
    elements.append(Paragraph(f"Oral Health Score: {score}/100 - {rating}", normal_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Main X-ray image
    elements.append(Paragraph("Full X-Ray Analysis", heading_style))
    
    # Convert base64 images to ReportLab images
    try:
        original_img = decode_base64_image(report_data['original_image'])
        annotated_img = decode_base64_image(report_data['annotated_image'])
        
        # Add the annotated X-ray image
        elements.append(pil_to_reportlab_image(annotated_img))
    except Exception as e:
        print(f"[ERROR] Error processing X-ray images: {str(e)}")
        elements.append(Paragraph("Error processing X-ray images", normal_style))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # Findings by category
    elements.append(Paragraph("Detailed Findings", heading_style))
    
    # Create a table for disease summary
    disease_data = [["Condition", "Number of Teeth", "Severity"]]
    for disease, teeth in report_data['teeth_by_disease'].items():
        severity = DISEASE_SEVERITY.get(disease, "Unknown")
        severity_text = "High" if severity >= 3 else "Medium" if severity >= 2 else "Low"
        disease_data.append([disease, len(teeth), severity_text])
    
    disease_table = Table(disease_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    disease_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(disease_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Detailed section for each disease category
    for disease, teeth in report_data['teeth_by_disease'].items():
        if len(teeth) > 0:
            elements.append(Paragraph(f"{disease} ({len(teeth)} teeth)", subheading_style))
            
            # Add recommendation for this disease
            if disease in DISEASE_RECOMMENDATIONS:
                elements.append(Paragraph(f"Recommendation: {DISEASE_RECOMMENDATIONS[disease]}", normal_style))
            
            # Add some sample tooth images for this category (limit to 3 per category to keep report size reasonable)
            sample_teeth = teeth[:3]
            
            # Create a table for the sample tooth images
            if sample_teeth:
                tooth_images = []
                for tooth in sample_teeth:
                    if 'image' in tooth:
                        try:
                            print(f"Processing tooth image for disease: {disease}")
                            img = decode_base64_image(tooth['image'])
                            # Use the same function signature as defined above
                            rl_img = pil_to_reportlab_image(img, width=1.75*inch)
                            tooth_images.append(rl_img)
                        except Exception as e:
                            print(f"Error processing tooth image: {e}")
                            # Create a blank placeholder image
                            blank_img = BytesIO()
                            Image.new('RGB', (100, 100), color=(200, 200, 200)).save(blank_img, format='JPEG')
                            blank_img.seek(0)
                            tooth_images.append(ReportLabImage(blank_img, width=1.5*inch))
                
                if tooth_images:
                    # Create a row of images
                    elements.append(Spacer(1, 0.1*inch))
                    image_table = Table([tooth_images], colWidths=[2*inch] * len(tooth_images))
                    image_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    elements.append(image_table)
            
            elements.append(Spacer(1, 0.15*inch))
    
    # Recommendations section
    elements.append(Paragraph("Recommendations", heading_style))
    for recommendation in recommendations:
        elements.append(Paragraph(f"• {recommendation}", normal_style))
        elements.append(Spacer(1, 0.05*inch))
    
    # Disclaimer
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("Disclaimer", subheading_style))
    elements.append(Paragraph(
        "This report is generated based on AI analysis and should be reviewed by a dental professional. "
        "It is not a substitute for professional dental advice, diagnosis, or treatment.",
        normal_style
    ))
    
    # Build the PDF
    doc.build(elements)
    
    return output_path
